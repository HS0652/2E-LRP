from test_GA import test_GA
from test_SA import test_SA
from test_GWO import test_GWO

file_list = ['instance/instance_5_15_100.json']

for file in file_list:
    test_GA(file)
    test_SA(file)
    test_GWO(file)
