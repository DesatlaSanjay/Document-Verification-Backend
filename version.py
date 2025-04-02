# import sys
# import platform

# def print_python_info():
#     print("Python interpreter:", sys.executable)
#     print("Python version:", platform.python_version())

# if __name__ == "__main__":
#     print_python_info()
try:
    import pyzbar
    print("pyzbar is installed.")
except ImportError:
    print("pyzbar is not installed.")
