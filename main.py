import sys

from modules.processing.processor import normalize_json_dataset, DATASET, create_processed_excel_files


def menu():
    print("1 - normalize_json_dataset\n2 - create_processed_excel_files")
    option: str = input("Enter your option: ")
    if option == "1":
        normalize_json_dataset(DATASET)
    elif option == "2":
        limit: int = 0
        try:
            limit = int(input("enter limit (default: 0): "))
        except ValueError:
            limit = 0
        except EOFError:
            sys.exit()
        create_processed_excel_files(limit)
    else:
        print("invalid option. Exiting...")



def main():
    menu()

if __name__ == "__main__":
    main()