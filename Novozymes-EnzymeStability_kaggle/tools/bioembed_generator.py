"""
https://github.com/sacdallago/bio_embeddings

"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder


def nespcsvloader(csv_file):
    '''
    Convert CSV into required Data
    '''
    datadf = pd.read_csv(csv_file)
    x = []; y = []
    for i in range(len(datadf)):
        x.append(datadf.loc[i,'protein_sequence'])
        y.append(datadf.loc[i,'seq_id'])
    return x, y

def create_bioembed_hdf5():

    print("Loading Model...")
    embedder = ProtTransBertBFDEmbedder()
    print("Loading Data...")
    seqs, ids = nespcsvloader("datasets/test.csv")

    hdf_file = h5py.File("datasets/Nesp-Test-bioembed.hdf5", "w-")

    for sid, seq in tqdm(zip(ids, seqs)):
        dset = hdf_file.create_dataset( str(sid), (1, 1024),
                                    dtype=np.float32,
                                    chunks=(1, 1024),)
        embed = embedder.embed(seq)
        embed = embedder.reduce_per_protein(embed)
        dset[:,:] = embed

    hdf_file.close()


def run_single_sequence(sequence=""):
    if not sequence:
        sequence = "MALLHSARVLSGVASAFHPGLAAAASARASSWWAHVEMGPPDPILGVTEAYKRDTNSKKMNLGVGAYRDDNGKPYVLPSVRKAEAQIAAKGLDKEYLPIGGLAEFCRASAELALGENSEVVKSGRFVTVQTISGTGALRIGASFLQRFFKFSRDVFLPKPSWGNHTPIFRDAGMQLQSYRYYDPKTCGFDFTGALEDISKIPEQSVLLLHACAHNPTGVDPRPEQWKEIATVVKKRNLFAFFDMAYQGFASGDGDKDAWAVRHFIEQGINVCLCQSYAKNMGLYGERVGAFTVICKDADEAKRVESQLKILIRPMYSNPPIHGARIASTILTSPDLRKQWLQEVKGMADRIIGMRTQLVSNLKKEGSTHSWQHITDQIGMFCFTGLKPEQVERLTKEFSIYMTKDGRISVAGVTSGNVGYLAHAIHQVTK"
    embedder = ProtTransBertBFDEmbedder()
    print('Loaded Model')
    embed = embedder.embed(seq)
    embed = embedder.reduce_per_protein(embed)
    print('Embed Generation Done!')
    print(embed.shape)
    np.savetxt('embedcheck.txt'.format(i), embed, delimiter =', ')


if __name__ == "__main__":

    create_bioembed_hdf5()

