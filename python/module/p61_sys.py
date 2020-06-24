import sys
print(sys.path)

# from test_import import p62_import

# p62_import.sum2()

print(f"================================")

from test_import.p62_import import sum2
sum2()