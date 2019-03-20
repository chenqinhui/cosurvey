middle = open('preProcessFinished/middle.txt', 'rb')
good = open('preProcessFinished/good.txt', 'rb')
bad = open('preProcessFinished/good.txt', 'rb')
out_file = open('preProcessFinished/comments.txt','a', encoding='utf-8')

for line in middle.readlines():
    out = str(line,encoding='utf-8')[:-2]+'\n'
    out_file.write(out)
for line in good.readlines():
    out = str(line,encoding='utf-8')
    out_file.write(out)
for line in bad.readlines():
    out = str(line, encoding='utf-8')
    out_file.write(out)
middle.close()
good.close()
bad.close()
out_file.close()
