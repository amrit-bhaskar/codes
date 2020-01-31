from tika import parser

raw = parser.from_file('sample.pdf')
f=open('file.txt','w')
f.write(raw['content'])
f.close()
#print(raw['content'])
