import os

file = open("report.md","w")
file.write("# Sample Report 6")
file.write("![platform image](https://raw.githubusercontent.com/timrizzi81/v2_demos/master/Platform%20Visual.png)")
file.close()

os.system('git config --global user.email "' + os.environ['GIT_EMAIL'] + '"')
os.system('git config --global user.name "'+ os.environ['GIT_USERNAME'] +'"')

os.system("git add .")
os.system("git commit -m 'Automated git commit from run'")
os.system("git pull origin master")
os.system("git push origin master")