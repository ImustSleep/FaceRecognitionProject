import pandas as pd
import os 

#gets the names of the videos in the video folder to add to types later
#mp4_folder_path = "../data/MyTrainSet/Workable_set/videos"
#video_name_list = [file[1:-4] for file in os.listdir(mp4_folder_path)]

df = pd.read_csv("../Data/sample_submission.csv")

df["image"] = df["image"].astype("string")
df["label_name"] = df["label_name"].astype("string")


faulty_name_corrector = {
    "alif":"akif",
    "alpre":"alper",
    "asper":"alper",
    "floarian":"florian",
    "lesse":"lasse",
    "mattias":"matthias",
    "nille":"nelli"
    }




#this goes over all the names, splits them and puts them into lists
#it also checks if the last character is a semicolon and if so removes it and goes further
#and it also puts an empty ist if the entry is nothing
def format_names(str_input):
    to_return = []
    if(str_input.lower() == "nothing"):
        return to_return
    else:
        if(str_input[-1] != ";"):
            to_return = str_input.lower().replace(" ","").split(";")
            for i in range(len(to_return)):
                if(to_return[i] in faulty_name_corrector):
                    to_return[i] = faulty_name_corrector[to_return[i]]
        else:
            to_return = str_input[:-1].lower().replace(" ","").split(";")
            for i in range(len(to_return)):
                if(to_return[i] in faulty_name_corrector):
                    to_return[i] = faulty_name_corrector[to_return[i]]
    return to_return
            

df["names"] =  df["label_name"].apply(
    format_names
    )

df["amount"] =  df["names"].apply(
    lambda x: len(x)
    )

# df["type"] =  df["image"].apply(
#     lambda x: "video" if x in video_name_list else "image"
#     )

df["image"] =  df["image"].apply(
     lambda x: x.zfill(4)
    )

df.to_csv("../Data/Cleanup/clean_dataset.csv", index=False)

name_counts = {}

counter = 0
for entry in  df["names"].dropna():
    counter +=1
    for name in entry:
        if(name == ""):
            raise Exception("Found the empty space on row: " + str(counter)) 
        if name in name_counts:
            name_counts[name] += 1
        else:
            name_counts[name] = 1