#This file is used to keep all the constants that i need for the project.


PATH = '/home/gargano/dataset/dataWithoutMasks'

USERS = ['Amparore', 'Baccega', 'Basile', 'Beccuti', 'Botta', 'Castagno', 'Davide', 'DiCaro', 'DiNardo','Esposito','Francesca','Giovanni','Gunetti','Idilio','Ines','Malangone','Maurizio','Michael','MirkoLai','MirkoPolato','Olivelli','Pozzato','Riccardo','Rossana','Ruggero','Sapino','Simone','Susanna','Theseider','Thomas']
USERS_EXCLUDED =['Amparore']

#A lambda function that filter and exclude the n-th user of user in the for loop
USERS_TRAINING = list(filter(lambda x: x not in USERS_EXCLUDED, USERS))