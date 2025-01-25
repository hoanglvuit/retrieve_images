
import os

path1 = "Humans\\1 (16185).jpg"  # Escaped backslashes
path2 = r"Humans\1 (1685).jpg" # Raw string literal
path3 = "Humans/1 (1685).jpg"  # Forward slashes

print(os.path.exists(path1)) #check if path exists
print(os.path.exists(path2)) #check if path exists
print(os.path.exists(path3)) #check if path exists