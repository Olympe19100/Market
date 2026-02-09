import os, json, zipfile
files = [f for f in os.listdir(".") if f.endswith("_ob200.data.zip")]
if not files:
    print("No files")
    exit()

z = zipfile.ZipFile(files[0])
f = z.open(z.namelist()[0])
line = f.readline()
d = json.loads(line)
print("Inner Keys:", list(d['data'].keys()))
print("Sample Inner:", str(d['data'])[:100])
