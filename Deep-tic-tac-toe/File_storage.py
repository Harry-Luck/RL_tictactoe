import os
import pickle

def save_mct(mct, path, game_name):
    filename = os.path.join(path, "mct_" + game_name + ".txt")
    with open(filename, "wb") as fp:   #Pickling
        pickle.dump(mct, fp)
    print("save_mct(): Monte Carlo Tree saved correctly to file: ", filename)

def load_mct(path, game_name):
    filename = os.path.join(path, "mct_" + game_name + ".txt")
    if os.path.isfile(filename):
        # load existing file
        print("load_mct(): Found existing Monte Carlo Tree file")
        with open(filename, "rb") as fp:   # Unpickling
            mct = pickle.load(fp)
    else:
        print("load_mct(): Creating new Monte Carlo Tree.")
        mct = {}
    return mct
