import os
import zipfile

os.system("pwd")
os.system("wget -P /output http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip")
os.system("wget -P /output http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip")

def unZipFile(unZipSrc,targeDir):
    if not os.path.isfile(unZipSrc):
        raise Exception('unZipSrc not exists:{0}'.format(unZipSrc))

    if not os.path.isdir(targeDir):
        os.makedirs(targeDir)

    print('unzip:{0}'.format(unZipSrc))

    unZf = zipfile.ZipFile(unZipSrc,'r')

    for name in unZf.namelist() :
        unZfTarge = os.path.join(targeDir,name)

        if unZfTarge.endswith("/"):
            #empty dir
            splitDir = unZfTarge[:-1]
            if not os.path.exists(splitDir):
                os.makedirs(splitDir)
        else:
            splitDir,_ = os.path.split(targeDir)

            if not os.path.exists(splitDir):
                os.makedirs(splitDir)

            hFile = open(unZfTarge,'wb')
            hFile.write(unZf.read(name))
            hFile.close()

    print('finish unzip!Destination dir:{0}'.format(targeDir))
    unZf.close()    

unZipFile('/output/Flickr8k_Dataset.zip','/output/Flickr8k_Dataset/')
unZipFile('/output/Flickr8k_text.zip','/output/Flickr8k_text/')

os.system("rm /output/Flickr8k_Dataset.zip")
os.system("rm /output/Flickr8k_text.zip")

