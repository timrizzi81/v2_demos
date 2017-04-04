import os

file = open("report.md","w")
file.write("# Sample Report")
file.close()

print("test")

os.system("git add .")
os.system("git commit -m 'testing automated git commit'")
os.system("git pull origin master")
os.system("git push origin master")
