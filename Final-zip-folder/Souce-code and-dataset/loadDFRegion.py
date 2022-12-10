import pickle

def getDF():
    # This pickle file is stored using pandas==1.4.1
    #picklefile = open('df_ByRegion.pkl', 'rb')
    #unpickle the dataframe
    #df = pickle.load(picklefile)
    #close file
    #picklefile.close()

    #main
    picklefile = open('df_ByRegion.pkl', 'rb')
    df = pickle.load(picklefile)
    picklefile.close()
    
    #get all regions
    dfRegionList=[]
    for i in range(1,11):
        if(i!=7):
            print(f"Grabbing region {i} dataframe...")
            picklefile = open(f'df_r{i}.pkl', 'rb')
            dfRegion = pickle.load(picklefile)
            dfRegionList.append(dfRegion)
            picklefile.close()
    return df, dfRegionList