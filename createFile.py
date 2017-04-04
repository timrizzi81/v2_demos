import os

file = open("report.md","w")
file.write("# Sample Report 2")
file.close()

os.system("git add .")
os.system("git commit -m 'Automated git commit from run'")
os.system("git pull origin master")
os.system("git push origin master")
