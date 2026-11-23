from functions.get_files_info import get_files_info


if __name__ == "__main__":
    print("Result for current directory:")
    print("  " + get_files_info("calculator", ".").replace("\n", "\n  "))

    print("Result for 'pkg' directory:")
    print("  " + get_files_info("calculator", "pkg").replace("\n", "\n  "))

    print("Result for '/bin' directory:")
    print(f"    {get_files_info('calculator', '/bin')}")

    print("Result for '../' directory:")
    print(f"    {get_files_info('calculator', '../')}")
