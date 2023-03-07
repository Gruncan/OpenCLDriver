
import os

total = 1000

def main():

    for i in range(total):
        v  = os.popen('./testbench -v').readlines()

    
        if i % 10 == 0:
            print(f"Tested {i}/{total} times!")

        for k in v:
            if not k.startswith("Computed '64/64' correct values in thread"):
                print("Failed after", i, "runs with line:")
                print(k)
                exit()
                return "Fail"
    return "Success"

f = main()
print(f)
