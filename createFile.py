import os

file = open("report.md","w")
file.write("# Sample Report")
file.close()

os.system("git add .")
os.system("git commit -m 'testing automated git commit'")
os.system("git push origin master")
