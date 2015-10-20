import os

os.system('python temporal_lobe_masks.py')
os.system('python PandoraPreProcessing_3T.py')
os.system('python PandoraPreProcessing_7T.py')
os.system('python 3T_Encoding_AllParticipants_AllRuns.py')
os.system('python 7T_Encoding_AllParticipants_AllRuns.py')
os.system('python Get_r2_of_voxels.py')
os.system('python MakePlots.py')
