import requests
import time
import json


class WikidataBatchFetcher:
    def __init__(self, email="my_email@example.com"):
        self.headers = {"User-Agent": f"SmartBatchBot/1.0 ({email})"}
        self.api_url = "https://www.wikidata.org/w/api.php"

    def _request_json(self, params, max_retries=4, timeout=120):
        """
        Retry Wikidata requests with timeout to avoid silent data loss from transient network issues.
        """
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(
                    self.api_url,
                    params=params,
                    headers=self.headers,
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_error = exc
                wait_sec = min(2 * attempt, 10)
                print(
                    f"   Request failed (attempt {attempt}/{max_retries}): {exc}. "
                    f"Retrying in {wait_sec}s..."
                )
                time.sleep(wait_sec)

        raise RuntimeError(f"Wikidata request retries exhausted: {last_error}")

    def _chunk_list(self, data, size=50):
        for i in range(0, len(data), size):
            yield data[i : i + size]

    def _is_important_claim(self, statement, prop_label):
        """
        [Core filtering logic] Determine whether a claim is important
        """
        mainsnak = statement.get("mainsnak", {})
        datatype = mainsnak.get("datatype")

        # 1. Core filter: absolutely exclude external database IDs
        if datatype == "external-id":
            return False

        # 2. Media file filter: if you only generate plain text, you can remove images and math formulas
        if datatype in ["commonsMedia", "math"]:
            # return False # Uncomment this line if you need image URLs
            pass

        # 3. Label filter: some properties are not external-id type but are still IDs (e.g. 'semantic scholar author ID')
        # If the property name contains 'identifier' or 'ID', it's usually noise
        if prop_label and ("identifier" in prop_label.lower() or " ID" in prop_label):
            return False

        return True

    def fetch_labels_map(self, id_list):
        """Batch fetch labels for IDs"""
        id_map = {}
        unique_ids = list(set(id_list))

        for chunk in self._chunk_list(unique_ids, 50):
            ids_str = "|".join(chunk)
            params = {
                "action": "wbgetentities",
                "ids": ids_str,
                "props": "labels",
                "languages": "en",
                "format": "json",
            }
            try:
                data = self._request_json(
                    params=params,
                    max_retries=4,
                    timeout=45,
                )
                if "entities" in data:
                    for eid, entity in data["entities"].items():
                        label = entity.get("labels", {}).get("en", {}).get("value", eid)
                        id_map[eid] = label
            except Exception as e:
                print(f"   Label fetch failed, skipping this chunk: {e}")
            time.sleep(0.5)
        return id_map

    def process_entities_batch(self, entity_ids):
        results = []
        total = len(entity_ids)
        print(f"🚀 Starting smart processing of {total} entities...")

        for i, chunk in enumerate(self._chunk_list(entity_ids, 50)):
            print(f"   Processing batch {i + 1}...")

            # 1. Fetch raw data
            ids_str = "|".join(chunk)
            params = {
                "action": "wbgetentities",
                "ids": ids_str,
                "props": "labels|descriptions|claims",
                "languages": "en",
                "format": "json",
            }
            data = self._request_json(
                params=params,
                max_retries=4,
                timeout=120,
            )

            # 2. Pre-scan: collect IDs and build preliminary property name mapping
            # We need to know what Pxxx is (e.g. P214 is 'VIAF ID') before deciding whether to discard it
            # So we need to collect Property IDs first
            ids_to_resolve = set()
            raw_entities = []

            if "entities" not in data:
                print(f"   Batch {i + 1} has no valid entity data")
                continue

            for eid, entity in data["entities"].items():
                if "missing" in entity:
                    print(f"   Batch {i + 1} entity {eid} is missing")
                    continue
                if "redirects" in entity:
                    print(f"   Batch {i + 1} entity {eid} is a redirect entity")
                    entity["id"] = entity["redirects"]["from"]
                raw_entities.append(entity)

                claims = entity.get("claims", {})
                for pid in claims.keys():
                    ids_to_resolve.add(pid)  # Add property IDs first (P641, P214...)

                    for stmt in claims[pid]:
                        # Collect Value IDs (Q2736...)
                        mainsnak = stmt.get("mainsnak", {})
                        if mainsnak.get("datatype") == "wikibase-item":
                            datavalue = mainsnak.get("datavalue")
                            if datavalue and "value" in datavalue:
                                value_id = datavalue["value"].get("id")
                                if value_id:
                                    ids_to_resolve.add(value_id)

            # 3. Batch translate all IDs (including property names and property values)
            label_map = self.fetch_labels_map(list(ids_to_resolve))

            # 4. Assemble and filter
            for entity in raw_entities:
                readable_data = {
                    "id": entity["id"],
                    "label": entity.get("labels", {}).get("en", {}).get("value", "N/A"),
                    "description": entity.get("descriptions", {}).get("en", {}).get("value", "N/A"),
                    "properties": {},
                }

                claims = entity.get("claims", {})
                for pid, stmts in claims.items():
                    prop_name = label_map.get(pid, pid)  # Get property name, e.g. "sport" or "VIAF ID"

                    # --- Filtering logic called here ---
                    # If any statement passes the filter, we want this property
                    # For simplicity, we check the first statement's datatype, since all statements under the same P ID share the same type
                    if not stmts:
                        print(f"   Batch {i + 1} entity {entity['id']} property {prop_name} has no valid statement")
                        continue

                    if not self._is_important_claim(stmts[0], prop_name):
                        print(f"   Batch {i + 1} entity {entity['id']} property {prop_name} filtered out")
                        continue  # Skip this property, it's noise

                    values = []
                    for stmt in stmts:
                        mainsnak = stmt.get("mainsnak", {})
                        if "datavalue" not in mainsnak:
                            continue

                        dv = mainsnak["datavalue"]
                        dtype = mainsnak.get("datatype")

                        if dtype == "wikibase-item":
                            tid = dv["value"]["id"]
                            values.append(label_map.get(tid, tid))
                        elif dtype == "time":
                            # Extract year or full date
                            time_str = dv["value"]["time"]
                            # Simple cleanup: +1888-00-00T00... -> 1888
                            if "-00-00T" in time_str:
                                values.append(time_str[1:5])
                            else:
                                values.append(time_str)
                        elif dtype in ["string", "monolingualtext"]:
                            val = dv["value"]
                            if isinstance(val, dict):
                                val = val.get("text", "")
                            values.append(val)
                        elif dtype == "quantity":
                            values.append(dv["value"]["amount"])

                    if values:
                        readable_data["properties"][prop_name] = values

                results.append(readable_data)

            time.sleep(1)  # API courtesy

        return results

# ================= Usage Example =================

if __name__ == "__main__":
    # Suppose you have 5 IDs to query (you can actually put up to 5000)
    target_ids = [
        "Q5311995",  # Dudley Town F.C.
        "Q18656",  # Manchester United
        "Q95",  # Google
        "Q5284",  # Bill Gates
        "Q42",  # Douglas Adams
    ]

    fetcher = WikidataBatchFetcher(email="researcher@university.edu")

    # Execute batch fetch
    start_time = time.time()
    final_data = fetcher.process_entities_batch(target_ids)
    end_time = time.time()

    print(f"\n✅ Done! Time elapsed: {end_time - start_time:.2f}s")

    # Print one of the results to inspect
    print(json.dumps(final_data[0], indent=2, ensure_ascii=False))

    # Save to file
    with open("wikidata_batch_results.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
