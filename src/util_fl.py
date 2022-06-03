#This file is used to keep all the constants that i need for the project.


PATH = '/home/gargano/dataset/dataWithoutMasks'

CATEGORIES = ["c00","c01","c02","c03","c04","c05","c06","c07","c08","c09","c10","c11","c12","c13","c14"]
NUMBER_CLASSES = 15
USERS = ['Amparore', 'Baccega', 'Basile', 'Beccuti', 'Botta', 'Castagno', 'Davide', 'DiCaro', 'DiNardo','Esposito','Francesca','Giovanni','Gunetti','Idilio','Ines','Malangone','Maurizio','Michael','MirkoLai','MirkoPolato','Olivelli','Pozzato','Riccardo','Rossana','Ruggero','Sapino','Simone','Susanna','Theseider','Thomas']
USERS_EXCLUDED =['Botta']

#A lambda function that filter and exclude the n-th user of user in the for loop
USERS_TRAINING = list(filter(lambda x: x not in USERS_EXCLUDED, USERS))
