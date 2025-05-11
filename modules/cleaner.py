import ast
import json
import logging

DATASET: str = "../dataset/beeradvocate.json"
DATASET_OUTPUT: str = "../dataset/beeradvocate-final.json"
LOGGER: logging.Logger = logging.getLogger(__name__)

def reorganize_dataset() -> dict:
    result: dict = {}

    with open(DATASET) as f:
        counter: int = 0
        for line in f:
            json_transform: dict = {}
            invalid_json = ast.literal_eval(line)
            try :
                json_transform["name"] = invalid_json["beer/name"]
                json_transform["brewer_id"] = invalid_json["beer/brewerId"]
                json_transform["abv"] = invalid_json["beer/ABV"]
                json_transform["style"] = invalid_json["beer/style"]
                json_transform["appearance"] = invalid_json["review/appearance"]
                json_transform["aroma"] = invalid_json["review/aroma"]
                json_transform["palate"] = invalid_json["review/palate"]
                json_transform["taste"] = invalid_json["review/taste"]
                json_transform["overall"] = invalid_json["review/overall"]
                json_transform["text"] = invalid_json["review/text"]
                json_transform["time"] = invalid_json["review/time"]
                json_transform["profile_name"] = invalid_json["review/profileName"]
            except KeyError:
                LOGGER.warning(f"Invalid Entry at line {counter}")
                continue
            result[counter] = json_transform.copy()
            counter+=1
    return result

def write_reorganized_json(result: dict) -> None:
    with open(DATASET_OUTPUT, "w") as f:
        f.write(json.dumps(result, indent=4))

def main():
    result: dict = reorganize_dataset()
    write_reorganized_json(result)

if __name__ == "__main__":
    main()