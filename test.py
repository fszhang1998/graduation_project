
file=open('val_gopro_gamma.list','r+')
for i in file:
    temp=i
    temp=temp.replace('sharp','blur')

    print(i)




