import os
import random
import sys
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from scipy.interpolate import interp1d
from scipy.stats import f, gmean, mannwhitneyu, multivariate_normal, ttest_1samp, ttest_ind, ttest_rel, wilcoxon
from sklearn.mixture import GaussianMixture

from movie_decoding.dataloader.patients import Patients
from movie_decoding.param.param_data import LABELS


def read_annotation(annotation_file: str) -> List[float]:
    annotation_path = os.path.dirname(__file__) + "/annotations"
    data = pd.read_csv(
        os.path.join(annotation_path, annotation_file),
        sep="^([^\s]*)\s",
        engine="python",
        header=None,
    )
    data[1] = pd.to_numeric(data[1], errors="coerce")
    data.dropna(subset=[1], inplace=True)
    surrogate_window = np.floor(data[1]).astype(int)
    surrogate_window = surrogate_window.tolist()

    return surrogate_window


annotations = [
    "562_FR1",
    "562_FR2",
    "563_FR1",
    "563_FR2",
    "566_FR1",
    "566_CR1",
    "566_FR2",
    "566_CR2",
    "567_FR1",
    "567_CR1",
    "567_FR2",
    "567_CR2",
    "568_FR1",
    "568_CR1",
    "572_FR1",
    "572_CR1",
    "572_FR2",
    "572_CR2",
    "i728_FR1a",
    "i728_FR1b",
    "i728_CR1",
    "i728_FR2",
    "i728_CR2",
]
surrogate_windows = defaultdict(list)

for annotation in annotations:
    surrogate_windows[annotation] = read_annotation(f"{annotation}.ann")

# fmt: off

# 2023-06-08 define 2nd value to concept exactly and will use only that one
patients = Patients()
patients.add_experiment(patient_id="555", experiment_name="free_recall1")
patients["555"]["free_recall1"]["attacks/bomb/bus/explosion"] = [35914, 93905]
patients["555"]["free_recall1"]["attacks/bomb/bus/explosion"].description = "including negotiation...more commonly what they said"
patients["555"]["free_recall1"]["hostage/exchange/sacrifice/negotiations"] = [9563, 22429, 28464, 47244, 54959, 62789, 71765, 101644]
patients["555"]["free_recall1"]["Jack Bauer"] = [12589, 61419, 106804, 112053]
# main terrorist
patients["555"]["free_recall1"]["Abu Fayed"] = [31634, 45969, 75902, 99888, 122944]
patients["555"]["free_recall1"]["Ahmed Amar"] = [118014]  # kid
patients["555"]["free_recall1"]["President"] = [5725, 51909]

# p562, exp 5.
patients.add_experiment(patient_id="562", experiment_name="free_recall1")
patients["555"]["free_recall1"]["LA"] = [47894]
patients["555"]["free_recall1"]["attacks/bomb/bus/explosion"] = [16662, 29149, 42753, 79223, 94616, 106387, 150762, 154253, 219886]
patients["555"]["free_recall1"]["CIA/FBI"] = [9763, 210852, 273087, 300741]
patients["562"]["free_recall1"]["hostage/exchange/sacrifice"] = [ 53874, 70537, 124121, 207739, 308822, ]
patients["562"]["free_recall1"]["handcuff/chair/tied"] = [259365]
patients["562"]["free_recall1"]["Jack Bauer"] = [ 62337, 74338, 137758, 139410, 223847, 248686, 286911, 321887]
patients["562"]["free_recall1"]["Abu Fayed"] = [ 133129, 140493, 233770, 263705, 286149, 346257]
patients["562"]["free_recall1"]["Ahmed Amar"] = [ 191496, 333707, 337780, 343262, 350226]
patients["562"]["free_recall1"]["President"] = [23949]

# p562, exp 7
patients.add_experiment(patient_id="562", experiment_name="free_recall2")

patients["562"]["free_recall2"]["LA"] = [50117]
patients["562"]["free_recall2"]["attacks/bomb/bus/explosion"] = [ 19270, 45668, 49602, 75683, 199756, 239419, 317987]
patients["562"]["free_recall2"]["white house/DC"] = [60295]
patients["562"]["free_recall2"]["CIA/FBI"] = [62511, 105390, 258610, 284353]
patients["562"]["free_recall2"]["hostage/exchange/sacrifice"] = [ 94940, 111449, 139963, 118219, 146279, 160157, 174136, 212141]
patients["562"]["free_recall2"]["Jack Bauer"] = [ 100545, 114639, 123671, 138993, 151212, 153702, 155272, 156697, 183406, 208166, 210566]
patients["562"]["free_recall2"]["Abu Fayed"] = [165285, 177856, 214729, 270081]
patients["562"]["free_recall2"]["Ahmed Amar"] = [251342, 263739, 269196]
patients["562"]["free_recall2"]["President"] = [53679, 282278]

# p563, Exp 10
patients.add_experiment(patient_id="563", experiment_name="free_recall1")

patients["563"]["free_recall1"]["attacks/bomb/bus/explosion"] = [17948]
patients["563"]["free_recall1"]["CIA/FBI"] = [34833, 181732, 205157, 221412, 246762]
patients["563"]["free_recall1"]["hostage/exchange/sacrifice/martyr"] = [107116, 139963, 299632]
patients["563"]["free_recall1"]["Jack Bauer"] = [104836, 124938, 134708, 223847, 263527, 278167]
patients["563"]["free_recall1"]["Chloe"] = [50858]
patients["563"]["free_recall1"]["Bill"] = [118876]
patients["563"]["free_recall1"]["Abu Fayed"] = [208077, 228022, 237282, 274762]
patients["563"]["free_recall1"]["Ahmed Amar"] = [167473, 182602, 194092]
patients["563"]["free_recall1"]["President"] = [58163, 63165, 91180]

free_recall_windows_563_FR1 = [
    [],  # LA
    [17948],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [34833, 181732, 205157, 221412, 246762],  # CIA/FBI
    [107116, 139963, 299632],  # hostage/exchange/sacrifice/martyr
    [],  # handcuff/chair/tied
    [104836, 124938, 134708, 223847, 263527, 278167],  # Jack Bauer
    [50858],  # Chloe
    [118876],  # Bill
    [208077, 228022, 237282, 274762],  # Abu Fayed (main terrorist)
    [167473, 182602, 194092],  # Ahmed Amar (kid)
    [58163, 63165, 91180],  # President
]

# p563, Exp 12
free_recall_windows_563_FR2 = [
    [],  # LA
    [9870, 24960],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [57000, 76653, 135626, 165960, 228360],  # CIA/FBI
    [172402, 183053, 200508, 252889],  # hostage/exchange/sacrifice
    [218790],  # handcuff/chair/tied
    [
        166527,
        183570,
        197798,
        203470,
        251550,
        262637,
        272425,
        280108,
        290238,
        301530,
    ],  # Jack Bauer
    [],  # Chloe
    [],  # Bill
    [145580, 221910, 250350, 255895, 263374, 298432],  # Abu Fayed (main terrorist)
    [63060, 69570],  # Ahmed Amar (kid)
    [85319, 128346],  # President
]

# p564, exp 3
free_recall_windows_564_FR1 = [
    [110429, 157973],  # LA
    [12064, 17366, 35598, 108439],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [43720, 47634],  # CIA/FBI
    [
        97212,
        136012,
        186583,
    ],  # hostage/exchange/sacrifice (including negotiation...more commonly what they said)
    [154512, 163743],  # handcuff/chair/tied
    [
        83685,
        128231,
        133557,
        153427,
        170833,
        178043,
        181438,
        189123,
        236780,
        238545,
    ],  # Jack Bauer
    [55194, 201640, 204830],  # Chloe
    [165258, 169078, 237995],  # Bill
    [120224, 124124, 127956, 141783],  # Abu Fayed (main terrorist)
    [],  # Ahmed Amar (kid)
    [65662, 93706, 208315, 213488],  # President
]

# p564, Exp 5
free_recall_windows_564_FR2 = [
    [2308],  # LA
    [8673, 16720, 53178, 110881, 114771],  # attacks/bomb/bus/explosion
    [83441],  # white house/DC
    [63304, 235019],  # CIA/FBI
    [223381, 238059],  # hostage/exchange/sacrifice
    [199782, 231391],  # handcuff/chair/tied
    [
        118506,
        129104,
        139320,
        148855,
        168405,
        198367,
        206866,
        217996,
        289288,
        296563,
    ],  # Jack Bauer
    [75761, 234356, 258988],  # Chloe
    [190338, 239774, 250218],  # Bill
    [108591, 133145],  # Abu Fayed (main terrorist)
    [],  # Ahmed Amar (kid)
    [82496, 89826, 136825, 216796, 248324],  # President
]

# p565, exp 6
free_recall_windows_565_FR1 = [
    [67455, 74175, 108990],  # LA
    [4519],  # attacks/bomb/bus/explosion
    [227120],  # white house/DC
    [145655],  # CIA/FBI
    [],  # hostage/exchange/sacrifice
    [],  # handcuff/chair/tied
    [24829, 179910, 233590],  # Jack Bauer
    [223345, 228095],  # Chloe
    [],  # Bill
    [],  # Abu Fayed (main terrorist)
    [],  # Ahmed Amar (kid)
    [],  # President
]

# p565, exp 8
free_recall_windows_565_FR2 = [
    [64621, 121530, 159526, 179880, 303209, 329610],  # LA
    [59250, 116640, 257613],  # attacks/bomb/bus/explosion
    [252954],  # white house/DC
    [],  # CIA/FBI
    [140070, 147540],  # hostage/exchange/sacrifice
    [],  # handcuff/chair/tied
    [23970, 135551],  # Jack Bauer
    [],  # Chloe
    [],  # Bill
    [],  # Abu Fayed (main terrorist)
    [],  # Ahmed Amar (kid)
    [],  # President
]

# p566, Exp 7 free recall
free_recall_windows_566_FR1 = [
    [],  # LA
    [19680, 42709],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [],  # CIA/FBI
    [54737, 89670, 101477, 180725, 279089],  # hostage/exchange/sacrifice
    [235425],  # handcuff/chair/tied
    [
        16510,
        55356,
        68015,
        78712,
        100120,
        117136,
        193986,
        206820,
        224710,
        250371,
        263760,
        278970,
    ],  # Jack Bauer
    [],  # Chloe
    [],  # Bill
    [24190, 174441, 196404, 209220],  # Abu Fayed (main terrorist)
    [135330, 148170, 157860, 177839, 189184, 214261],  # Ahmed Amar (kid)
    [7878, 48987],  # President
]

# p566, Exp 7 cued recall
free_recall_windows_566_CR1 = [
    [242093],  # LA
    [187130, 236610, 271189, 366764],  # attacks/bomb/bus/explosion
    [463945],  # white house/DC
    [40331, 461605, 467597, 503405, 522132],  # CIA/FBI
    [
        36374,
        151800,
        181740,
        228104,
        303372,
        363247,
        417022,
        494434,
    ],  # hostage/exchange/sacrifice
    [
        345172,
        357764,
        387810,
        396268,
        404770,
        410571,
        428764,
        436569,
    ],  # handcuff/chair/tied
    [
        4934,
        26904,
        32845,
        53553,
        216072,
        221191,
        302160,
        312600,
        318530,
        344745,
        352170,
        361366,
        375642,
        388110,
        396090,
        406890,
        412800,
        419056,
        426493,
        438630,
        536346,
    ],  # Jack Bauer
    [28945, 51983, 60260, 509034],  # Chloe
    [],  # Bill
    [
        92005,
        144240,
        285912,
        291158,
        297776,
        307333,
        317590,
        327078,
        368423,
        390960,
        434610,
        441689,
        490350,
        519359,
        534002,
    ],  # Abu Fayed (main terrorist)...calls him "Indu"
    [
        86379,
        100395,
        106426,
        119420,
        138696,
        156971,
        322300,
        482898,
        489930,
    ],  # Ahmed Amar (kid)
    [175356, 194610, 203766, 212220, 222252, 465096],  # President
]

# p566, Exp 9 free recall
free_recall_windows_566_FR2 = [
    [89026],  # LA
    [80971, 146633, 153628, 442781],  # attacks/bomb/bus/explosion
    [7340],  # white house/DC
    [168772, 175068, 275872, 539737, 545742, 575922, 605874, 615624],  # CIA/FBI
    [
        386983,
        427140,
        450253,
        499243,
        517650,
        680875,
        751676,
    ],  # hostage/exchange/sacrifice
    [412550, 660575, 673250, 791073],  # handcuff/chair/tied
    [
        36977,
        74164,
        236531,
        387448,
        406041,
        412430,
        418460,
        424308,
        450797,
        460271,
        474942,
        485645,
        499500,
        564411,
        644249,
        666272,
        674291,
        681096,
        687488,
        702071,
        715010,
        780883,
        791478,
        808130,
        819887,
    ],  # Jack Bauer (calls him Kai/Kite)
    [217637, 223420, 231768, 550686, 618514, 637550],  # Chloe
    [],  # Bill
    [
        505471,
        511930,
        525840,
        531443,
        572697,
        579158,
        589007,
        599850,
        682904,
        716014,
        722036,
        746304,
        752562,
        770731,
        797900,
        804242,
    ],  # Abu Fayed (main terrorist ANDU)
    [
        268810,
        291966,
        346811,
        358945,
        730820,
        736850,
        743834,
        753931,
        760160,
    ],  # Ahmed Amar (kid)
    [14340, 58170, 64020],  # President
]

# p566, Exp 9 cued recall
free_recall_windows_566_CR2 = [
    [],  # LA
    [140430, 177659],  # attacks/bomb/bus/explosion
    [314031],  # white house/DC
    [2495, 72150, 210051, 311773, 327041, 332190],  # CIA/FBI
    [147883, 262816, 280328],  # hostage/exchange/sacrifice
    [285638],  # handcuff/chair/tied
    [
        14400,
        149869,
        170773,
        170773,
        195180,
        254640,
        261324,
        277350,
        283989,
    ],  # Jack Bauer
    [1674, 13139, 237199],  # Chloe
    [320788],  # Bill
    [
        45150,
        171034,
        123396,
        171034,
        198771,
        208429,
    ],  # Abu Fayed (main terrorist...thinks he's ASAAD)
    [73796],  # Ahmed Amar (kid)
    [120053, 157410],  # President
]

# p567, Exp 8 free recall. Note he remembers Ahmed as guy you never see...
# so was careful to only include Fayed references for "the terrorist" doing the negotiations
free_recall_windows_567_FR1 = [
    [15204],  # LA
    [3546, 9036, 21573],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [],  # CIA/FBI
    [100980, 246225],  # hostage/exchange/sacrifice
    [206850],  # handcuff/chair/tied
    [
        43170,
        45930,
        51056,
        59820,
        76260,
        96072,
        124230,
        133500,
        165082,
        173280,
        199200,
        204643,
        238524,
    ],  # Jack Bauer
    [75025, 89860, 133740],  # Chloe
    [237480, 243060],  # Bill
    [117267, 150146, 167197, 180407],  # Abu Fayed (main terrorist)
    [],  # Ahmed Amar (kid)
    [37140, 228269],  # President
]

# p567, Exp 8 cued recall
free_recall_windows_567_CR1 = [
    [],  # LA
    [178266],  # attacks/bomb/bus/explosion
    [306150],  # white house/DC
    [7383, 35760, 59684, 181620, 202020, 208020],  # CIA/FBI/DHS/US Government
    [244140],  # hostage/exchange/sacrifice
    [232500, 240566],  # handcuff/chair/tied
    [
        2024,
        154380,
        204450,
        227160,
        234004,
        241216,
        249060,
        256560,
        264550,
        302848,
        317000,
    ],  # Jack Bauer
    [1080, 6394],  # Chloe
    [],  # Bill
    [170730, 197850],  # Abu Fayed (main terrorist)
    [64320, 73230],  # Ahmed Amar (kid)
    [143403],  # President
]

# p567, Exp 10 FR2
free_recall_windows_567_FR2 = [
    [],  # LA
    [9180, 16085, 256516],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [114562, 135120, 202405],  # CIA/FBI
    [168962],  # hostage/exchange/sacrifice
    [192730],  # handcuff/chair/tied
    [
        80145,
        86456,
        94230,
        102690,
        143413,
        150460,
        160716,
        169623,
        177263,
        187955,
        228397,
        237058,
        273503,
        278562,
    ],  # Jack Bauer
    [141188, 147554, 157899, 165785, 199922],  # Chloe
    [],  # Bill
    [90766, 214470, 241958, 248370, 254186, 282059],  # Abu Fayed (main terrorist)
    [314579, 319720],  # Ahmed Amar (kid)
    [44550, 61530, 73992],  # President
]

# p567, Exp 10 CR2
free_recall_windows_567_CR2 = [
    [],  # LA
    [],  # attacks/bomb/bus/explosion
    [277410],  # white house/DC
    [5346, 276126],  # CIA/FBI
    [103560, 136593, 203910],  # hostage/exchange/sacrifice
    [209760],  # handcuff/chair/tied
    [
        100290,
        137523,
        175710,
        185042,
        191460,
        198760,
        205860,
        214780,
        224971,
        256675,
    ],  # Jack Bauer
    [],  # Chloe
    [],  # Bill
    [32541, 37708, 119428, 132957, 140594, 153993],  # Abu Fayed (main terrorist)
    [8820, 30730, 36937, 143737, 274350],  # Ahmed Amar (kid)
    [62201, 96914],  # President
]

# p568, Exp 5 FR1
free_recall_windows_568_FR1 = [
    [],  # LA
    [5400, 122910, 130101],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [],  # CIA/FBI
    [41721],  # hostage/exchange/sacrifice
    [],  # handcuff/chair/tied
    [30690, 39810],  # Jack Bauer
    [],  # Chloe
    [],  # Bill
    [],  # Abu Fayed (main terrorist)
    [],  # Ahmed Amar (kid)
    [],  # President
]

# p568, Exp 5 CR1
free_recall_windows_568_CR1 = [
    [],  # LA
    [110557],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [15151, 91770, 339916, 347157],  # CIA/FBI
    [285881, 330894],  # hostage/exchange/sacrifice
    [272209],  # handcuff/chair/tied
    [
        17736,
        22762,
        233668,
        241090,
        271695,
        277351,
        282914,
        289741,
        329685,
        356813,
    ],  # Jack Bauer
    [13414],  # Chloe
    [],  # Bill
    [207806, 213743, 219350, 227707, 240813, 347625],  # Abu Fayed (main terrorist)
    [101360, 106616],  # Ahmed Amar (kid)
    [135007],  # President
]

# # p572, Exp 10 FR1
free_recall_windows_572_FR1 = [
    [],  # LA
    [21243, 43995],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [175981],  # CIA/FBI
    [102625, 110574, 147493, 197735],  # hostage/exchange/sacrifice
    [],  # handcuff/chair/tied
    [113032, 136548, 141325, 145122, 153029, 223104, 228311, 232767],  # Jack Bauer
    [196430],  # Chloe
    [],  # Bill
    [119796, 204677, 219329],  # Abu Fayed (main terrorist)
    [],  # Ahmed Amar (kid)
    [63832],  # President
]

# # p572, Exp 10 CR1
free_recall_windows_572_CR1 = [
    [94610, 136372, 312235],  # LA
    [],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [78115, 331085],  # CIA/FBI
    [],  # hostage/exchange/sacrifice
    [243450, 272014],  # handcuff/chair/tied
    [
        21672,
        197628,
        203850,
        243450,
        248952,
        259173,
        265739,
        276773,
        281485,
    ],  # Jack Bauer
    [6677],  # Chloe
    [],  # Bill
    [156150, 183389, 195611, 208522, 213789],  # Abu Fayed (main terrorist)
    [66293],  # Ahmed Amar (kid)
    [],  # President
]

# # p572, Exp 13 FR2
free_recall_windows_572_FR2 = [
    [28240],  # LA
    [7470, 27361],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [125600, 260670],  # CIA/FBI
    [156030, 170512],  # hostage/exchange/sacrifice
    [185512],  # handcuff/chair/tied
    [
        141721,
        156750,
        161760,
        167580,
        172863,
        180618,
        185838,
        209504,
        218484,
        230121,
        236002,
    ],  # Jack Bauer
    [260227],  # Chloe
    [],  # Bill
    [],  # Abu Fayed (main terrorist)
    [],  # Ahmed Amar (kid)
    [41462, 64033],  # President
]

# # p572, Exp 13 CR2
free_recall_windows_572_CR2 = [
    [],  # LA
    [],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [3660, 10770, 16889, 86790],  # CIA/FBI
    [],  # hostage/exchange/sacrifice
    [254610, 261068, 270540],  # handcuff/chair/tied
    [
        22749,
        34980,
        203096,
        209730,
        217067,
        248850,
        260278,
        265710,
        270870,
        276780,
    ],  # Jack Bauer
    [595, 7764, 20340, 32974],  # Chloe
    [],  # Bill
    [183720, 194366, 211576],  # Abu Fayed (main terrorist)
    [78403, 85247],  # Ahmed Amar (kid)
    [120774, 128669, 151560, 160832],  # President
]

# i702, Exp 046
free_recall_windows_i702_FR1 = [
    [],  # LA
    [19951, 72590],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [58791, 178410],  # CIA/FBI
    [136710],  # hostage/exchange/sacrifice
    [],  # handcuff/chair/tied
    [30120, 41760, 128651, 137040, 146047, 164626],  # Jack Bauer
    [170680],  # Chloe
    [],  # Bill
    [24466, 40683],  # Abu Fayed (main terrorist)
    [],  # Ahmed Amar (kid)
    [],  # President
]

# i702, Exp 048
free_recall_windows_i702_FR2 = [
    [],  # LA
    [],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [20180, 31200],  # CIA/FBI
    [56567],  # hostage/exchange/sacrifice
    [],  # handcuff/chair/tied
    [47802, 53350, 65005],  # Jack Bauer
    [],  # Chloe
    [],  # Bill
    [38562, 51235, 194750],  # Abu Fayed (main terrorist)
    [147343, 157864, 169073, 175574, 182430, 188565, 205784],  # Ahmed Amar (kid)
    [],  # President
]

# 429000
offset_i728 = ((55 * 60 + 45) - (48 * 60 + 36)) * 1000
# i728, Exp 45
free_recall_windows_i728_FR1a = [
    [124558, 417700],  # LA
    [13424, 24000, 50557, 316980],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [199230, 208440],  # CIA/FBI
    [80203, 107320],  # hostage/exchange/sacrifice
    [117541, 382823],  # handcuff/chair/tied
    [
        62170,
        84994,
        103440,
        142080,
        150217,
        179967,
        194678,
        217477,
        223770,
        231774,
        248430,
        353005,
        361650,
        370760,
        385950,
        398146,
    ],  # Jack Bauer
    [],  # Chloe
    [],  # Bill
    [
        138692,
        163526,
        170310,
        177390,
        182730,
        241517,
        247070,
        326094,
        394199,
        420926,
    ],  # Abu Fayed (main terrorist)
    [286350, 304287, 312510, 320400],  # Ahmed Amar (kid)
    [],  # President
]

# i728, Exp 46 # patient did a second free recall before sleep!
free_recall_windows_i728_FR1b = [
    [206067, 322500],  # LA
    [4891, 20927, 36013, 92160, 109980],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [60630, 240480, 309000, 357990],  # CIA/FBI
    [100460, 156930],  # hostage/exchange/sacrifice
    [209310],  # handcuff/chair/tied
    [
        73590,
        98070,
        114090,
        119469,
        131550,
        139241,
        159981,
        168676,
        175678,
        183849,
        197370,
        209640,
        215610,
        225240,
        251970,
        264600,
        276540,
        291390,
    ],  # Jack Bauer
    [],  # Chloe
    [],  # Bill
    [
        53663,
        82067,
        218841,
        296044,
        306599,
        464034,
        480602,
    ],  # Abu Fayed (main terrorist)
    [
        350250,
        355200,
        376331,
        385918,
        391050,
        409250,
        426330,
        434124,
        446656,
        460350,
        467970,
        473436,
    ],  # Ahmed Amar (kid)
    [],  # President
]

free_recall_windows_i728_FR1 = [
    fr1a + [fr1b_item + offset_i728 for fr1b_item in fr1b]
    for fr1a, fr1b in zip(free_recall_windows_i728_FR1a, free_recall_windows_i728_FR1b)
]
surrogate_windows_i728_FR1 = surrogate_windows["i728_FR1a"] + [
    fr1b_item + offset_i728 for fr1b_item in surrogate_windows["i728_FR1b"]
]

# i728, Exp 46
free_recall_windows_i728_CR1 = [
    [291690],  # LA
    [64362],  # attacks/bomb/bus/explosion
    [328410, 462510],  # white house/DC
    [2280, 321780, 459420, 466108],  # CIA/FBI
    [128250, 253026],  # hostage/exchange/sacrifice
    [239880],  # handcuff/chair/tied
    [
        8916,
        119366,
        127470,
        136860,
        155736,
        208135,
        239019,
        247266,
        255986,
        267715,
        472470,
    ],  # Jack Bauer
    [1360, 457724, 471787, 478483],  # Chloe
    [],  # Bill
    [175666, 194935, 210254],  # Abu Fayed (main terrorist)
    [79583],  # Ahmed Amar (kid)
    [120390, 135540, 154625, 160680],  # President
]

# i728, Exp 50
free_recall_windows_i728_FR2 = [
    [28590, 232280, 440880],  # LA
    [4870, 20223, 27840, 66600, 451710],  # attacks/bomb/bus/explosion
    [],  # white house/DC
    [272070, 305430, 317880, 475740, 500310],  # CIA/FBI
    [60744, 81600, 95564, 124740, 176656],  # hostage/exchange/sacrifice
    [239640, 419790],  # handcuff/chair/tied
    [
        63107,
        73980,
        79746,
        89137,
        96457,
        104100,
        124140,
        132734,
        177145,
        187080,
        193680,
        203183,
        218190,
        240330,
        249254,
        273420,
        282690,
        313290,
        327450,
        346590,
        353250,
        363129,
        374280,
        383730,
        391410,
        405750,
        420660,
        609310,
    ],  # Jack Bauer
    [259081],  # Chloe
    [],  # Bill
    [
        53268,
        155534,
        292560,
        340740,
        351690,
        361530,
        603336,
        613840,
    ],  # Abu Fayed (main terrorist)
    [
        474450,
        501990,
        519120,
        533370,
        549010,
        560693,
        575220,
        586431,
        594630,
        618970,
    ],  # Ahmed Amar (kid)
    [114000, 186030, 257460],  # President
]

# i728, Exp 50
free_recall_windows_i728_CR2 = [
    [310560],  # LA
    [156120, 197400, 305109],  # attacks/bomb/bus/explosion
    [322500],  # white house/DC
    [73678, 325539],  # CIA/FBI
    [],  # hostage/exchange/sacrifice
    [],  # handcuff/chair/tied
    [
        10860,
        18193,
        125261,
        131070,
        144810,
        157410,
        164670,
        187358,
        203357,
        213960,
        240885,
        251820,
        258570,
        275777,
    ],  # Jack Bauer
    [1771],  # Chloe
    [],  # Bill
    [181358, 199348, 205560, 211024],  # Abu Fayed (main terrorist)
    [70110, 78447],  # Ahmed Amar (kid)
    [126630, 133200, 141120, 146940, 153240],  # President
]

# fmt: on


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[[i + np.argmin(s[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global max of dmax-chunks of locals max
    lmax = lmax[[i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax


def _remove_na_single(x, axis="rows"):
    """Remove NaN in a single array.
    This is an internal Pingouin function.
    """
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
    else:
        # 2D arrays
        ax = 1 if axis == "rows" else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
    # Check if missing values are present
    if ~x_mask.all():
        ax = 0 if axis == "rows" else 1
        ax = 0 if x.ndim == 1 else ax
        x = x.compress(x_mask, axis=ax)
    return x


def remove_na(x, y=None, paired=False, axis="rows"):
    # Safety checks
    x = np.asarray(x)
    assert axis in ["rows", "columns"], "axis must be rows or columns."

    if y is None:
        return _remove_na_single(x, axis=axis)
    elif isinstance(y, (int, float, str)):
        return _remove_na_single(x, axis=axis), y
    else:  # y is list, np.array, pd.Series
        y = np.asarray(y)
        assert y.size != 0, "y cannot be an empty list or array."
        # Make sure that we just pass-through if y have only 1 element
        if y.size == 1:
            return _remove_na_single(x, axis=axis), y
        if x.ndim != y.ndim or paired is False:
            # x and y do not have the same dimension
            x_no_nan = _remove_na_single(x, axis=axis)
            y_no_nan = _remove_na_single(y, axis=axis)
            return x_no_nan, y_no_nan

    # At this point, we assume that x and y are paired and have same dimensions
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
        y_mask = ~np.isnan(y)
    else:
        # 2D arrays
        ax = 1 if axis == "rows" else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
        y_mask = ~np.any(np.isnan(y), axis=ax)

    # Check if missing values are present
    if ~x_mask.all() or ~y_mask.all():
        ax = 0 if axis == "rows" else 1
        ax = 0 if x.ndim == 1 else ax
        both = np.logical_and(x_mask, y_mask)
        x = x.compress(both, axis=ax)
        y = y.compress(both, axis=ax)
    return x, y


def multivariate_ttest(X, Y=None, paired=False):
    from scipy.stats import f

    x = np.asarray(X)
    assert x.ndim == 2, "x must be of shape (n_samples, n_features)"

    if Y is None:
        y = np.zeros(x.shape[1])
        # Remove rows with missing values in x
        x = x[~np.isnan(x).any(axis=1)]
    else:
        nx, kx = x.shape
        y = np.asarray(Y)
        assert y.ndim in [1, 2], "Y must be 1D or 2D."
        if y.ndim == 1:
            # One sample with specified null
            assert y.size == kx
        else:
            # Two-sample
            err = "X and Y must have the same number of features (= columns)."
            assert y.shape[1] == kx, err
            if paired:
                err = "X and Y must have the same number of rows if paired."
                assert y.shape[0] == nx, err
        # Remove rows with missing values in both x and y
        x, y = remove_na(x, y, paired=paired, axis="rows")

    # Shape of arrays
    nx, k = x.shape
    ny = y.shape[0]
    # assert nx >= 5, "At least five samples are required."

    if y.ndim == 1 or paired is True:
        n = nx
        if y.ndim == 1:
            # One sample test
            cov = np.cov(x, rowvar=False)
            diff = x.mean(0) - y
        else:
            # Paired two sample
            cov = np.cov(x - y, rowvar=False)
            diff = x.mean(0) - y.mean(0)
        inv_cov = np.linalg.pinv(cov, hermitian=True)
        t2 = (diff @ inv_cov) @ diff * n
    else:
        n = nx + ny - 1
        x_cov = np.cov(x, rowvar=False)
        y_cov = np.cov(y, rowvar=False)
        pooled_cov = ((nx - 1) * x_cov + (ny - 1) * y_cov) / (n - 1)
        # inv_cov = np.linalg.pinv((1 / nx + 1 / ny) * pooled_cov, hermitian=True)
        diff = x.mean(0) - y.mean(0)
        # t2 = (diff @ inv_cov) @ diff
        t2 = (nx * ny) / (nx + ny) * np.matmul(np.matmul(diff.transpose(), np.linalg.inv(pooled_cov)), diff)

    # F-value, degrees of freedom and p-value
    fval = t2 * (n - k) / (k * (n - 1))
    # df1 = k
    # df2 = n - k
    # pval = f.sf(fval, df1, df2)
    F = f(k, n - k)
    pval = 1 - F.cdf(fval)

    # Create output dictionnary
    return pval


def getRandomIdxWithAcceptableBinsBack(mask_bins, max_bins_back, activation_width, max_attempts=100):
    max_bins_back_plus_width = max_bins_back + activation_width[-1]  # farthest bin backwards acceptable for random idx

    attempts = 0
    while attempts < max_attempts:
        # Randomly select an index from the mask
        selected_index = random.choice(mask_bins)

        if selected_index > max_bins_back_plus_width:  # otherwise will run out of values at beginning
            # Generate a range of indices from selected_index - bb to selected_index
            required_indices = set(range(selected_index - max_bins_back + 1, selected_index + 1))

            # Check if all required indices are in the mask
            if required_indices.issubset(set(mask_bins)):
                return selected_index
        attempts += 1  # Increment the number of attempts

    # If the loop exits without finding a suitable index, raise an error
    raise ValueError("Unable to find a suitable index within the specified number of attempts.")


def hotelling_t_squared(X, Y):
    # Convert the input to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    n, k = Y.shape
    if X.ndim == 1 and Y.ndim == 2:
        # Case 1: X is a vector, and Y is a matrix
        vector = X
        matrix = Y
    elif X.ndim == 2 and Y.ndim == 2:
        # Case 2: Both X and Y are matrices
        vector_mean = np.mean(X, axis=0)
        matrix = Y
    else:
        raise ValueError("Invalid input dimensions. X and Y must be either a vector and a matrix or two matrices.")

    # Calculate the sample means of the vector and matrix
    matrix_mean = np.mean(matrix, axis=0)

    # Calculate the sample covariance matrix of the matrix
    cov_matrix = np.cov(matrix, rowvar=False)

    # Calculate the inverse of the covariance matrix
    cov_matrix_inv = np.linalg.pinv(cov_matrix, hermitian=True)

    # Calculate the difference between the means
    mean_diff = vector_mean - matrix_mean

    # Calculate the Hotelling's T-squared statistic
    t_squared = (mean_diff @ cov_matrix_inv) @ mean_diff

    # Calculate the F-statistic
    f_statistic = t_squared * (n - k) / (k * (n - 1))

    # Calculate the p-value
    df1 = k
    df2 = n - k
    # p_value = 1 - f.cdf(f_statistic, df1, df2)
    p_value = f.sf(f_statistic, df1, df2)

    return t_squared, f_statistic, p_value


def getEmpiricalConceptPs(activations, free_recall_windows, bins_back=-4, activation_width=4):
    # Updated 2023-07-04 to mask out the free recall windows so the random permutations
    # don't select activations from them. Also without replacement now too but that shouldn't matter
    bin_size = 0.25
    permutations = 100000
    time_bins = np.arange(0, len(activations) * bin_size, bin_size)  # all the time bins

    concept_Ps = []  # empirical p-value for actual concept is greater than permutated samples
    for concept_i, concept_vocalizations in enumerate(free_recall_windows):  # for each concept
        if len(concept_vocalizations) > 0:  # if person said the concept at all
            target_activations = []
            target_activations_indices = []
            for concept_vocalization in concept_vocalizations:  # get the ranges for each mention
                # get the average activation score for these ranges
                closest_end = np.abs(time_bins - concept_vocalization / 1000).argmin()

                if (
                    closest_end - (-bins_back + activation_width) >= 0
                ):  # shouldn't happen that concepts are this close to beginning but just in case
                    # grab only those bins before concept mention
                    target_activations.extend(
                        activations[
                            closest_end - (-bins_back + activation_width) : closest_end - (-bins_back),
                            concept_i,
                        ]
                    )
                    target_activations_indices.extend(
                        np.arange(
                            closest_end - (-bins_back + activation_width),
                            closest_end - (-bins_back),
                        )
                    )
            # Create a mask to exclude the specified indices
            mask = np.ones(len(activations), dtype=bool)
            mask[target_activations_indices] = False

            # Compare to equivalent shuffles after excluding the mask
            avg_activations = np.mean(target_activations)
            sig_counter = []
            for perm in range(permutations):
                # get random indices of same length as temp_activations
                sampled_activations = np.random.choice(
                    activations[mask, concept_i],
                    size=len(target_activations),
                    replace=False,
                )
                if avg_activations > np.mean(sampled_activations):
                    sig_counter.append(1)
                else:
                    sig_counter.append(0)
            concept_Ps.append(1 - sum(sig_counter) / permutations)
        else:
            concept_Ps.append(np.nan)
    concept_Ps = np.round(concept_Ps, 5)
    # print results
    # sorted_idxs = np.argsort(concept_Ps)
    # labels = np.array(LABELS)[sorted_idxs] # sort labels and concepts
    # concept_Ps = concept_Ps[sorted_idxs]
    labels = np.array(LABELS)
    return concept_Ps, labels


def getEmpiricalConceptPs_yyding(
    activations,
    free_recall_windows,
    bins_back=-4,
    activation_width=4,
    surrogate_mask=[],
):
    # Updated 2024-02-14
    bin_size = 0.25
    permutations = 100000
    time_bins = np.arange(0, len(activations) * bin_size, bin_size)

    concept_Ps = []
    for concept_i, concept_vocalizations in enumerate(free_recall_windows):
        if len(concept_vocalizations) > 0:
            closest_ends = np.abs(time_bins - (np.array(concept_vocalizations) / 1000).reshape(-1, 1)).argmin(axis=1)
            valid_ends = closest_ends[closest_ends - (-bins_back + activation_width) >= 0]

            if len(valid_ends) > 0:
                range_indices = [np.arange(end - 16, end) for end in valid_ends]
                target_indices = [
                    np.arange(end - (-bins_back + activation_width), end - (-bins_back)) for end in valid_ends
                ]
                target_activations = activations[target_indices, concept_i].flatten()

                mask = np.ones(len(activations), dtype=bool)
                mask[np.concatenate(range_indices)] = False
                if len(surrogate_mask) > 0:
                    mask[surrogate_mask[0] : surrogate_mask[1]] = False

                avg_activation = np.mean(target_activations)
                random_indices = np.random.choice(np.where(mask)[0], (permutations, len(target_activations)))
                sampled_activations = activations[random_indices, concept_i]

                sig_counter = np.sum(np.mean(sampled_activations, axis=1) < avg_activation)
                concept_Ps.append(1 - sig_counter / permutations)
            else:
                concept_Ps.append(np.nan)
        else:
            concept_Ps.append(np.nan)

    concept_Ps = np.round(concept_Ps, 5)
    labels = np.array(LABELS)
    return concept_Ps, labels


# def getEmpiricalConceptPs_parallel(args):
#     # Updated 2023-07-04 to mask out the free recall windows so the random permutations
#     # don't select activations from them. Also without replacement now too but that shouldn't matter

#     bin_size=0.25
#     permutations=10000
#     activations, free_recall_windows, bins_back, activation_width, result_dict = args
#     bins_back = np.abs(bins_back)
# #     import scipy.stats as stats
# #     from scipy.stats import mannwhitneyu
#     time_bins = np.arange(0,len(activations)*bin_size, bin_size) # all the time bins

#     concept_Ps = [] # empirical p-value for actual concept is greater than permutated samples
#     for concept_i,concept_vocalizations in enumerate(free_recall_windows): # for each concept
#         if len(concept_vocalizations)>0: # if person said the concept at all
#             target_activations = []
#             target_activations_indices = []
#             for concept_vocalization in concept_vocalizations: # get the ranges for each mention
#                 # get the average activation score for these ranges
#                 closest_end = np.abs(time_bins-concept_vocalization/1000).argmin()

#                 if closest_end-(bins_back+activation_width)>=0: # shouldn't happen that concepts are this close to beginning but just in case
#                     # grab only those bins before concept mention
#                     target_activations.extend(activations[closest_end-(bins_back+activation_width):closest_end - bins_back, concept_i])
#                     target_activations_indices.extend(np.arange(closest_end-(bins_back+activation_width), closest_end - bins_back))
#             # Create a mask to exclude the specified indices
#             mask = np.ones(len(activations), dtype=bool)
#             mask[target_activations_indices] = False

#             # Compare to equivalent shuffles after excluding the mask
#             avg_activations = np.mean(target_activations)
#             sig_counter = []
#             for perm in range(permutations):
#                 # get random indices of same length as temp_activations
#                 sampled_activations = np.random.choice(activations[mask,concept_i], size=len(target_activations), replace=False)
#                 if avg_activations > np.mean(sampled_activations):
#                     sig_counter.append(1)
#                 else:
#                     sig_counter.append(0)
#             concept_Ps.append(1-sum(sig_counter)/permutations)
#         else:
#             concept_Ps.append(np.nan)
#     concept_Ps = np.round(concept_Ps,5)
#     # print results
#     # sorted_idxs = np.argsort(concept_Ps)
#     # labels = np.array(LABELS)[sorted_idxs] # sort labels and concepts
#     # concept_Ps = concept_Ps[sorted_idxs]
#     labels = np.array(LABELS)
#     result_dict[(bins_back, activation_width)] = (concept_Ps, labels)
#     #return concept_Ps, labels


def getEmpiricalConceptPs_hoteling(
    activations,
    free_recall_windows,
    bins_back,
    activation_width,
    bin_size=0.25,
    permutations=5000,
):
    time_bins = np.arange(0, len(activations) * bin_size, bin_size)  # all the time bins
    max_time_back = 4
    sig_vector_test = []
    for concept_i, concept_vocalizations in enumerate(free_recall_windows):  # for each concept
        if len(concept_vocalizations) > 0:  # if person said the concept at all
            target_activations = []
            target_activations_indices = []
            for concept_vocalization in concept_vocalizations:  # get the ranges for each concept mention
                # get the average activation score for each aw window prior to bb

                closest_end = np.abs(time_bins - concept_vocalization / 1000).argmin()
                # grab only those bins before concept mention
                temp_tas = []
                if (
                    closest_end - (np.max(np.abs(bins_back)) + np.max(activation_width)) >= 0
                ):  # if concept too close to beginning skip it
                    for aw in activation_width:
                        for bb in np.abs(bins_back):
                            temp_tas.append(
                                np.mean(
                                    activations[
                                        closest_end - (bb + aw) : closest_end - bb,
                                        concept_i,
                                    ]
                                )
                            )
                            target_activations_indices.extend(
                                np.arange(closest_end - (bb + aw), closest_end - bb)
                            )  # grab all the indices used in the average not just the bb start
                target_activations.append(temp_tas)
            # Create a mask to exclude the specified concept indices
            target_activations_indices = sorted(set(target_activations_indices))
            mask = np.ones(len(activations), dtype=bool)
            mask[target_activations_indices] = False  # remove these idxs from consideration
            start_indices = np.arange(0, np.max(np.abs(bins_back)) + np.max(activation_width))
            mask[start_indices] = False
            # now you have activations for given concept and a mask to avoid them for shuffle procedure.
            # Compare to equivalent shuffles after excluding the mask

            mask_bins = np.where(mask)[0]  # possible indices
            max_bins_back = int(max_time_back / bin_size)
            surrogate_activation_vectors = []
            for selected_index in mask_bins:
                # random_activations = []
                # for num_runs in range(len(concept_vocalizations)):  # mimic number of concepts to match distribution
                # selected_index = getRandomIdxWithAcceptableBinsBack(mask_bins, max_bins_back, activation_width)

                # for surrogate from given random time want to recreate the aw by bb process at that point
                temp_surrogate = []
                for aw in activation_width:
                    for bb in np.abs(bins_back):
                        if (
                            selected_index - (bb + aw) >= 0
                        ):  # should have already checked for acceptable bins back but keep JIC
                            temp_act = activations[
                                selected_index - (bb + aw) : selected_index - bb,
                                concept_i,
                            ]
                            temp_surrogate.append(np.mean(temp_act))
                # random_activations.append(temp_surrogate)
                surrogate_activation_vectors.append(temp_surrogate)
            # surrogate_activation_vectors.append(np.mean(random_activations, axis=0))
            # surrogate_activation_vectors.extend(random_activations)

            if np.size(target_activations) > 0:
                temp_p = multivariate_ttest(target_activations, surrogate_activation_vectors)
                # _, _, temp_p = hotelling_t_squared(target_activations, surrogate_activation_vectors)
                sig_vector_test.append(temp_p)
            else:
                sig_vector_test.append(np.nan)
        else:  # didn't say concept
            sig_vector_test.append(np.nan)

    concept_Ps = sig_vector_test
    concept_Ps = np.round(concept_Ps, 5)
    # print results
    # sorted_idxs = np.argsort(concept_Ps)
    # labels = np.array(LABELS)[sorted_idxs]  # sort labels and concepts
    # concept_Ps = concept_Ps[sorted_idxs]
    labels = np.array(LABELS)
    return concept_Ps, labels


def find_target_activation_indices(time, concept_vocalz_msec, win_range_bins, end_inclusive=True):
    concept_vocalz_bin = []
    target_activations_indices = []
    if len(concept_vocalz_msec) > 0:  # if person said the concept at all
        for concept_vocalization in concept_vocalz_msec:  # get the indices for each mention
            concept_vocalization_bin = np.abs(time - concept_vocalization / 1000).argmin()
            win_edge_1 = concept_vocalization_bin + win_range_bins[0]
            win_edge_2 = concept_vocalization_bin + win_range_bins[1]
            if end_inclusive:
                win_edge_2 += 1
            if win_edge_1 >= 0 and win_edge_2 < len(time):  # only take vocalizations where full window can be obtained
                target_activations_indices.append(np.arange(win_edge_1, win_edge_2, dtype=int))
                concept_vocalz_bin.append(concept_vocalization_bin)
    else:
        concept_vocalz_bin = []
        target_activations_indices = []
    return concept_vocalz_bin, target_activations_indices


def interp_threshold_crossing_points(y, x, threshold):
    # find mask where y is above threshold
    above_thresh_mask = y > threshold
    # change boolean to int to find direction of crossing
    crossing_mask_vals = above_thresh_mask.astype(int)
    crossing_dirs = np.diff(crossing_mask_vals)
    # remove zeros from crossing dirs
    crossing_dirs = crossing_dirs[crossing_dirs != 0]
    # find indices where crossing occurs
    above_threshold_indices = np.where(np.diff((above_thresh_mask)))[0]
    following_indices = above_threshold_indices + 1
    # find exact crossing points using interpolation
    x_interp = []
    for i, x1 in enumerate(above_threshold_indices):
        x2 = following_indices[i]
        cross_dir = crossing_dirs[i]
        if cross_dir == 1:
            x_interp1 = np.interp(threshold, y[x1 : x2 + 1], x[x1 : x2 + 1])
        elif cross_dir == -1:  # if crossing goes from above to below threshold flip the array
            x_interp1 = np.interp(threshold, np.flip(y[x1 : x2 + 1]), np.flip(x[x1 : x2 + 1]))
        x_interp.append(x_interp1)
    return x_interp


def find_area_above_threshold(y, x, threshold):
    # Interpolate the threshold crossing points to get exact crossing values
    x_interp = interp_threshold_crossing_points(y, x, threshold)

    # Add the interpolated crossing points to the data
    y = np.hstack((y, np.ones_like(x_interp) * threshold))
    x = np.hstack((x, x_interp))

    # Sort the data based on the x values
    sorted_inds = np.argsort(x)
    y = y[sorted_inds]
    x = x[sorted_inds]

    # Subtract the threshold from the data and set negative values to 0
    y = y - threshold
    y[y < 0] = 0

    # Calculate the area under the curve using the trapezoidal rule
    area = np.trapezoid(y, x=x)

    return area


def find_area_above_threshold_yyding(y, x, threshold, bin_range=[0, 8]):
    # lmin, lmax = hl_envelopes_idx(y)
    # yp = y[lmax]
    # xp = x[lmax]
    # y = np.interp(x, xp, yp)

    yy = y[bin_range[0] : bin_range[1]]
    xx = x[bin_range[0] : bin_range[1]]

    # area = sum([(max(n - threshold, 0) + min(n - threshold, 0)) for n in yy])
    x_interp = interp_threshold_crossing_points(yy, xx, threshold)
    yy = np.hstack((yy, np.ones_like(x_interp) * threshold))
    xx = np.hstack((xx, x_interp))
    sorted_inds = np.argsort(xx)
    yy = yy[sorted_inds]
    xx = xx[sorted_inds]
    yy = yy - threshold

    # yy = yy - threshold
    yy[yy < 0] = 0
    area = np.trapezoid(yy, x=xx)
    return area


def get_rand_idx_with_acceptable_window(mask_bins, win_range_bins, rng=None, max_attempts=100, end_inclusive=True):
    # based on JS code, edited to use win bins and np Generator instance for random number generation
    if rng is None:
        rng = np.random.default_rng()  # Use the default random number generator if not provided

    attempts = 0
    while attempts < max_attempts:
        # Randomly select an index from the mask
        selected_index = rng.choice(mask_bins)

        # Generate a range of indices from selected_index - bb to selected_index
        if end_inclusive:
            required_indices = set(
                range(
                    selected_index + win_range_bins[0],
                    selected_index + win_range_bins[1] + 1,
                )
            )
        else:
            required_indices = set(
                range(
                    selected_index + win_range_bins[0],
                    selected_index + win_range_bins[1],
                )
            )

        # Check if all required indices are in the mask
        if required_indices.issubset(set(mask_bins)):
            return selected_index, rng
        attempts += 1  # Increment the number of attempts
    return None, rng


def find_random_trial_indices(
    n_rand_trials,
    all_indicies,
    win_range_bins,
    bins_apart_threshold=5,
    rng=None,
    with_replacement=False,
):
    n_inds = 0
    n_to_find = n_rand_trials
    random_trial_inds = []
    while n_to_find > 0:
        for i in range(n_to_find):
            [idx, rng] = get_rand_idx_with_acceptable_window(all_indicies, win_range_bins, rng, end_inclusive=True)
            random_trial_inds.append(idx)

        if not with_replacement:
            random_trial_inds = np.unique(random_trial_inds)  # Remove duplicates

        # remove an index if the difference between two inds is below a set amount
        random_trial_inds = np.sort(random_trial_inds)
        ind_diff = np.diff(random_trial_inds)
        too_close_idx = ind_diff < bins_apart_threshold
        if np.any(too_close_idx):
            too_close_inds = [
                np.where(too_close_idx)[0][0],
                np.where(too_close_idx)[0][0] + 1,
            ]
            remove_ind = rng.choice(too_close_inds)  # randomly choose one of the too close inds to remove
            random_trial_inds = np.delete(random_trial_inds, remove_ind)
        random_trial_inds = random_trial_inds.tolist()
        n_inds = len(random_trial_inds)
        n_to_find = n_rand_trials - n_inds

    random_trial_indices = [
        np.arange(idx + win_range_bins[0], idx + win_range_bins[1] + 1, 1) for idx in random_trial_inds
    ]
    return random_trial_inds, random_trial_indices


def GMM_test_3segments(activations, free_recall_windows, bin_size=0.25, permutations=5000):
    time_bins = np.arange(0, len(activations) * bin_size, bin_size)
    max_bin_back = 16
    bin_back_segments = [(-12, 0), (-12, -4), (-16, 0)]
    use_vanilla_gmm = False
    for concept_i, concept_vocalizations in enumerate(free_recall_windows):
        if len(concept_vocalizations) > 0:  # if person said the concept at all
            target_activations = []
            target_activations_left = []
            target_activations_middle = []
            target_activations_right = []
            target_activations_indices = []
            for concept_vocalization in concept_vocalizations:  # get the ranges for each concept mention
                # get the average activation score for each aw window prior to bb

                closest_end = np.abs(time_bins - concept_vocalization / 1000).argmin()
                # grab only those bins before concept mention
                if closest_end - max_bin_back >= 0:  # if concept too close to beginning skip it
                    target_activations.append(activations[closest_end - max_bin_back + 1 : closest_end + 1, concept_i])
                    target_activations_left.append(
                        activations[
                            closest_end - (-bin_back_segments[0][0]) + 1 : closest_end - (-bin_back_segments[0][1]) + 1,
                            concept_i,
                        ]
                    )
                    target_activations_middle.append(
                        activations[
                            closest_end - (-bin_back_segments[1][0]) + 1 : closest_end - (-bin_back_segments[1][1]) + 1,
                            concept_i,
                        ]
                    )
                    target_activations_right.append(
                        activations[
                            closest_end - (-bin_back_segments[2][0]) + 1 : closest_end - (-bin_back_segments[2][1]) + 1,
                            concept_i,
                        ]
                    )
                    target_activations_indices.extend(
                        np.arange(closest_end - max_bin_back, closest_end)
                    )  # grab all the indices used in the average not just the bb start

            # Create a mask to exclude the specified concept indices
            target_activations_indices = sorted(set(target_activations_indices))
            mask = np.ones(len(activations), dtype=bool)
            mask[target_activations_indices] = False  # remove these idxs from consideration
            start_indices = np.arange(0, max_bin_back + 1)
            mask[start_indices] = False
            # now you have activations for given concept and a mask to avoid them for shuffle procedure.
            # Compare to equivalent shuffles after excluding the mask

            mask_bins = np.where(mask)[0]  # possible indices
            surrogate_activations = []
            surrogate_activations_left = []
            surrogate_activations_middle = []
            surrogate_activations_right = []
            # for selected_index in mask_bins:
            #     surrogate_activation_vectors.append(activations[selected_index - bin_back:selected_index, concept_i])
            for selected_index in mask_bins:
                # for perm in range(permutations):
                #     selected_index = random.choice(mask_bins)
                surrogate_activations.append(
                    activations[
                        selected_index - max_bin_back + 1 : selected_index + 1,
                        concept_i,
                    ]
                )
                surrogate_activations_left.append(
                    activations[
                        selected_index
                        - (-bin_back_segments[0][0])
                        + 1 : selected_index
                        - (-bin_back_segments[0][1])
                        + 1,
                        concept_i,
                    ]
                )
                surrogate_activations_middle.append(
                    activations[
                        selected_index
                        - (-bin_back_segments[1][0])
                        + 1 : selected_index
                        - (-bin_back_segments[1][1])
                        + 1,
                        concept_i,
                    ]
                )
                surrogate_activations_right.append(
                    activations[
                        selected_index
                        - (-bin_back_segments[2][0])
                        + 1 : selected_index
                        - (-bin_back_segments[2][1])
                        + 1,
                        concept_i,
                    ]
                )

            target_activations_left = np.array(target_activations_left)
            target_activations_middle = np.array(target_activations_middle)
            target_activations_right = np.array(target_activations_right)
            surrogate_activations_left = np.array(surrogate_activations_left)
            surrogate_activations_middle = np.array(surrogate_activations_middle)
            surrogate_activations_right = np.array(surrogate_activations_right)
            target_activations = np.array(target_activations)
            surrogate_activations = np.array(surrogate_activations)

            # """test"""
            # mm = surrogate_activations_middle.mean(-1)
            mm = gmean(surrogate_activations_middle, axis=-1)
            mmm = mm.mean()
            # plt.hist(surrogate_activations_middle.mean(-1))
            # plt.show()
            # plt.close()

            stddd = mm.std()
            thhhh = mmm - 1 * stddd
            maskkkk = mm > thhhh
            surrogate_activations_middle = surrogate_activations_middle[maskkkk]

            gmm_l = GaussianMixture(n_components=1)
            gmm_m = GaussianMixture(n_components=1)
            gmm_r = GaussianMixture(n_components=1)
            gmm_l.fit(surrogate_activations_left)
            gmm_m.fit(surrogate_activations_middle)
            gmm_r.fit(surrogate_activations_right)

            my_gmm_l = multivariate_normal(mean=gmm_l.means_[0], cov=gmm_l.covariances_[0])
            my_gmm_m = multivariate_normal(mean=gmm_m.means_[0], cov=gmm_m.covariances_[0])
            my_gmm_r = multivariate_normal(mean=gmm_r.means_[0], cov=gmm_r.covariances_[0])
            low = 5
            high = 95

            for i in range(len(target_activations_middle)):
                # left check
                if use_vanilla_gmm:
                    likelihood_concept_l = gmm_l.score_samples(target_activations_left[i].reshape(1, -1))[0]
                    likelihood_surrogate_l = gmm_l.score_samples(surrogate_activations_left)
                else:
                    likelihood_concept_l = my_gmm_l.logpdf(target_activations_left[i].reshape(1, -1))
                    likelihood_surrogate_l = my_gmm_l.logpdf(surrogate_activations_left)
                # left_threshold_likelihood_l = np.percentile(likelihood_surrogate_l, 100 - significance_level * 100)
                least_likely_threshold_l = np.percentile(likelihood_surrogate_l, low)
                within_tail_l = (
                    likelihood_concept_l <= least_likely_threshold_l
                )  # or likelihood_concept_l >= left_threshold_likelihood_l
                mask_most_likely_l = likelihood_surrogate_l >= np.percentile(
                    likelihood_surrogate_l, high
                )  # | (likelihood_surrogate_l <= np.percentile(likelihood_surrogate_l, 2))
                mask_least_likely_l = likelihood_surrogate_l <= np.percentile(likelihood_surrogate_l, low)

                # middle check
                if use_vanilla_gmm:
                    likelihood_concept_m = gmm_m.score_samples(target_activations_middle[i].reshape(1, -1))[0]
                    likelihood_surrogate_m = gmm_m.score_samples(surrogate_activations_middle)
                else:
                    likelihood_concept_m = my_gmm_m.logpdf(target_activations_middle[i].reshape(1, -1))
                    likelihood_surrogate_m = my_gmm_m.logpdf(surrogate_activations_middle)
                # left_threshold_likelihood_m = np.percentile(likelihood_surrogate_m, 100 - significance_level * 100)
                least_likely_threshold_m = np.percentile(likelihood_surrogate_m, low)
                within_tail_m = (
                    likelihood_concept_m <= least_likely_threshold_m
                )  # or likelihood_concept_m >= left_threshold_likelihood_m
                mask_most_likely_m = likelihood_surrogate_m >= np.percentile(likelihood_surrogate_m, high)
                mask_least_likely_m = likelihood_surrogate_m <= np.percentile(likelihood_surrogate_m, low)

                # right check
                if use_vanilla_gmm:
                    likelihood_concept_r = gmm_r.score_samples(target_activations_right[i].reshape(1, -1))[0]
                    likelihood_surrogate_r = gmm_r.score_samples(surrogate_activations_right)
                else:
                    likelihood_concept_r = my_gmm_r.logpdf(target_activations_right[i].reshape(1, -1))
                    likelihood_surrogate_r = my_gmm_r.logpdf(surrogate_activations_right)
                # left_threshold_likelihood_r = np.percentile(likelihood_surrogate_r, 100 - significance_level * 100)
                least_likely_threshold_r = np.percentile(likelihood_surrogate_r, low)
                within_tail_r = (
                    likelihood_concept_r <= least_likely_threshold_r
                )  # or likelihood_concept_r >= left_threshold_likelihood_r
                mask_most_likely_r = likelihood_surrogate_r >= np.percentile(
                    likelihood_surrogate_r, high
                )  # | (likelihood_surrogate_r <= np.percentile(likelihood_surrogate_r, 2))
                mask_least_likely_r = likelihood_surrogate_r <= np.percentile(likelihood_surrogate_r, low)

                is_significant = within_tail_l or within_tail_m or within_tail_r
                print(
                    f"{LABELS[concept_i]} {i}th incident is very unlikely to happen:",
                    [within_tail_l, within_tail_m, within_tail_r],
                )
                x = np.linspace(0, max_bin_back, max_bin_back) - max_bin_back
                if is_significant:
                    fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex="col")
                    axs[0].plot(x, target_activations[i], label=f"{LABELS[concept_i]} {i}th")
                    axs[0].set_ylabel("vocal activations")
                    axs[0].get_yaxis().set_label_coords(-0.08, 0.5)
                    axs[0].set_ylim([0, 1])
                    axs[0].axhline(y=np.max(target_activations[i]), c="green")

                    mask_most_likely_all = np.logical_or.reduce(
                        [mask_most_likely_l, mask_most_likely_m, mask_most_likely_r]
                    )
                    most_likely_indices = np.where(mask_most_likely_all)[0]
                    most_likely_samples = random.sample(list(most_likely_indices), 30)
                    for ml in most_likely_samples:
                        axs[1].plot(x, surrogate_activations[ml])
                        axs[1].set_ylabel("most likely")
                        axs[1].get_yaxis().set_label_coords(-0.08, 0.5)
                        axs[1].set_ylim([0, 1])
                    axs[1].axhline(y=np.max(surrogate_activations[mask_most_likely_all]), c="red")
                    axs[1].axhline(y=np.max(target_activations[i]), c="green")

                    mask_least_likely_all = np.logical_or.reduce(
                        [mask_least_likely_l, mask_least_likely_m, mask_least_likely_r]
                    )
                    least_likely_indices = np.where(mask_least_likely_all)[0]
                    least_likely_samples = random.sample(list(least_likely_indices), 30)
                    for ll in least_likely_samples:
                        axs[2].plot(x, surrogate_activations[ll])
                        axs[2].set_ylabel("least likely")
                        axs[2].get_yaxis().set_label_coords(-0.08, 0.5)
                        axs[2].set_ylim([0, 1])
                    axs[2].axhline(y=np.max(surrogate_activations[mask_least_likely_all]), c="red")
                    axs[2].axhline(y=np.max(target_activations[i]), c="green")

                    all_samples = random.sample(list(np.arange(0, len(surrogate_activations))), 30)
                    for jj in all_samples:
                        axs[3].plot(x, surrogate_activations[jj])
                        axs[3].set_ylabel("randomly selected", fontsize=10)
                        axs[3].get_yaxis().set_label_coords(-0.08, 0.5)
                        axs[3].set_ylim([0, 1])
                    axs[3].axhline(y=np.max(surrogate_activations), c="red")
                    axs[3].axhline(y=np.max(target_activations[i]), c="green")
                    xticks = np.arange(-16, 0 + 2, 2)
                    axs[3].set_xticks(xticks)
                    axs[3].set_xticklabels(map(int, xticks))
                    plt.suptitle(f"{LABELS[concept_i]} {i}th")
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                if LABELS[concept_i] == "Jack":
                    fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex="col")
                    axs[0].plot(x, target_activations[i], label=f"{LABELS[concept_i]} {i}th")
                    axs[0].set_ylabel("vocal activations")
                    axs[0].get_yaxis().set_label_coords(-0.08, 0.5)
                    axs[0].set_ylim([0, 1])
                    axs[0].axhline(y=np.max(target_activations[i]), c="green")

                    # mask_most_likely_all = np.logical_or.reduce([mask_most_likely_l, mask_most_likely_m, mask_most_likely_r])
                    most_likely_indices = np.where(mask_most_likely_m)[0]
                    most_likely_samples = random.sample(list(most_likely_indices), 30)
                    for ml in most_likely_samples:
                        axs[1].plot(x, surrogate_activations[ml])
                        axs[1].set_ylabel("most likely")
                        axs[1].get_yaxis().set_label_coords(-0.08, 0.5)
                        axs[1].set_ylim([0, 1])
                    axs[1].axhline(y=np.max(surrogate_activations[most_likely_indices]), c="red")
                    axs[1].axhline(y=np.max(target_activations[i]), c="green")

                    # mask_least_likely_all = np.logical_or.reduce([mask_least_likely_l, mask_least_likely_m, mask_least_likely_r])
                    least_likely_indices = np.where(mask_least_likely_m)[0]
                    least_likely_samples = random.sample(list(least_likely_indices), 30)
                    for ll in least_likely_samples:
                        axs[2].plot(x, surrogate_activations[ll])
                        axs[2].set_ylabel("least likely")
                        axs[2].get_yaxis().set_label_coords(-0.08, 0.5)
                        axs[2].set_ylim([0, 1])
                    axs[2].axhline(y=np.max(surrogate_activations[least_likely_indices]), c="red")
                    axs[2].axhline(y=np.max(target_activations[i]), c="green")

                    all_samples = random.sample(list(np.arange(0, len(surrogate_activations))), 30)
                    for jj in all_samples:
                        axs[3].plot(x, surrogate_activations[jj])
                        axs[3].set_ylabel("randomly selected", fontsize=10)
                        axs[3].get_yaxis().set_label_coords(-0.08, 0.5)
                        axs[3].set_ylim([0, 1])
                    axs[3].axhline(y=np.max(surrogate_activations), c="red")
                    axs[3].axhline(y=np.max(target_activations[i]), c="green")
                    xticks = np.arange(-16, 0 + 2, 2)
                    axs[3].set_xticks(xticks)
                    axs[3].set_xticklabels(map(int, xticks))
                    print(gmm_m.means_)
                    print(gmm_m.covariances_)
                    print("concept: ", likelihood_concept_m)
                    print(
                        "concept presentile: ",
                        np.searchsorted(np.sort(likelihood_surrogate_m), likelihood_concept_m)
                        / len(likelihood_surrogate_m)
                        * 100,
                    )
                    print(
                        "least likely threshold: ",
                        np.percentile(likelihood_surrogate_m, 10),
                    )
                    print(
                        "most likely threhold: ",
                        np.percentile(likelihood_surrogate_m, 90),
                    )
                    plt.suptitle(f"{LABELS[concept_i]} {i}th")
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                    mask1 = likelihood_surrogate_m >= np.percentile(likelihood_surrogate_m, 90)
                    mask2 = likelihood_surrogate_m <= np.percentile(likelihood_surrogate_m, 10)
                    plt.hist(surrogate_activations_middle[mask1].mean(-1))
                    # plt.hist(gmean(surrogate_activations_middle[mask1], axis=-1))
                    plt.xlim([0, 1])
                    plt.title("most likely 10%")
                    plt.xlabel("average")
                    plt.show()
                    plt.close()
                    plt.hist(surrogate_activations_middle[mask2].mean(-1))
                    # plt.hist(gmean(surrogate_activations_middle[mask2], axis=-1))
                    plt.xlim([0, 1])
                    plt.title("least likely 10%")
                    plt.xlabel("average")
                    plt.show()
                    plt.close()

                # mask_all = np.logical_or.reduce([mask_l, mask_m, mask_r])
                # if LABELS[concept_i] == 'CIA/FBI':
                #     present_2_indices = np.where(mask_m)[0]
                #     present_2_times = present_2_indices * 0.25
                #     real_times = np.array(concept_vocalizations) / 1000
                #     print(f"{LABELS[concept_i]} {i}th incident significant at: ", present_2_times)
                #     print(f"{LABELS[concept_i]} {i}th incident real significant at: ", real_times)

                # print(f"{LABELS[concept_i]} {i}th incident Belongs to 95% tail:", within_tail_l or within_tail_m or within_tail_r)


def GMM_test(activations, free_recall_windows, bin_size=0.25, permutations=5000):
    time_bins = np.arange(0, len(activations) * bin_size, bin_size)
    max_bin_back = 16
    # bin_back_segments = [(-12, 0), (-12, -4), (-16, 0)]
    possible_bins = list(range(-16, 1))
    min_length = 4
    bin_back_segments = []
    for start in range(len(possible_bins) - min_length + 1):
        for end in range(start + min_length, len(possible_bins) + 1):
            subinterval = possible_bins[start:end]
            bin_back_segments.append(subinterval)

    use_vanilla_gmm = False
    for concept_i, concept_vocalizations in enumerate(free_recall_windows):
        if len(concept_vocalizations) <= 0:  # if person said the concept at all
            continue
        for incident_i, concept_vocalization in enumerate(
            concept_vocalizations
        ):  # get the ranges for each concept mention
            # get the average activation score for each aw window prior to bb
            target_activations = []
            target_activations_segments = []
            target_activations_indices = []
            closest_end = np.abs(time_bins - concept_vocalization / 1000).argmin()
            # grab only those bins before concept mention
            if closest_end - max_bin_back < 0:
                continue
            target_activations.append(activations[closest_end - max_bin_back + 1 : closest_end + 1, concept_i])
            target_activations_indices.extend(
                np.arange(closest_end - max_bin_back, closest_end)
            )  # grab all the indices used in the average not just the bb start

            for interval in bin_back_segments:
                target_activations_segments.append(
                    activations[
                        closest_end - (-interval[0]) : closest_end - (-interval[-1]) + 1,
                        concept_i,
                    ]
                )

            # Create a mask to exclude the specified concept indices
            target_activations_indices = sorted(set(target_activations_indices))
            mask = np.ones(len(activations), dtype=bool)
            mask[target_activations_indices] = False  # remove these idxs from consideration
            start_indices = np.arange(0, max_bin_back + 1)
            mask[start_indices] = False
            # now you have activations for given concept and a mask to avoid them for shuffle procedure.
            # Compare to equivalent shuffles after excluding the mask

            mask_bins = np.where(mask)[0]  # possible indices
            surrogate_activations = []
            surrogate_activations_segments = []
            for interval in bin_back_segments:
                tmp_a = []
                tmp_as = []
                for selected_index in mask_bins:
                    # for perm in range(permutations):
                    #     selected_index = random.choice(mask_bins)
                    tmp_a.append(
                        activations[
                            selected_index - max_bin_back + 1 : selected_index + 1,
                            concept_i,
                        ]
                    )
                    tmp_as.append(
                        activations[
                            selected_index - (-interval[0]) : selected_index - (-interval[-1]) + 1,
                            concept_i,
                        ]
                    )
                surrogate_activations.append(tmp_a)
                surrogate_activations_segments.append(tmp_as)

            target_activations = np.array(target_activations)
            surrogate_activations = np.array(surrogate_activations)

            segment_significancy = []
            for segment_i in range(len(target_activations_segments)):
                target = np.array(target_activations_segments[segment_i])
                surrogate = np.array(surrogate_activations_segments[segment_i])
                # """test"""
                # # mm = surrogate_activations_middle.mean(-1)
                # mm = gmean(surrogate_activations_middle, axis=-1)
                # mmm = mm.mean()
                # # plt.hist(surrogate_activations_middle.mean(-1))
                # # plt.show()
                # # plt.close()

                # stddd = mm.std()
                # thhhh = mmm - 1*stddd
                # maskkkk = mm > thhhh
                # surrogate_activations_middle = surrogate_activations_middle[maskkkk]
                gmm = GaussianMixture(n_components=1)
                gmm.fit(surrogate)
                my_gmm = multivariate_normal(mean=gmm.means_[0], cov=gmm.covariances_[0])
                low = 5
                high = 95

                if use_vanilla_gmm:
                    likelihood_concept = gmm.score_samples(target.reshape(1, -1))[0]
                    likelihood_surrogate = gmm.score_samples(surrogate)
                else:
                    likelihood_concept = my_gmm.logpdf(target.reshape(1, -1))
                    likelihood_surrogate = my_gmm.logpdf(surrogate)

                least_likely_threshold = np.percentile(likelihood_surrogate, low)
                within_tail = likelihood_concept <= least_likely_threshold
                mask_most_likely = likelihood_surrogate >= np.percentile(likelihood_surrogate, high)
                mask_least_likely = likelihood_surrogate <= np.percentile(likelihood_surrogate, low)

                is_significant = within_tail
                # print(f"{LABELS[concept_i]} {incident_i}th incident is very unlikely to happen in segment {bin_back_segments[segment_i]}: ", within_tail)
                segment_significancy.append(is_significant)

            significant_indices = np.where(segment_significancy)[0]
            x = np.linspace(0, max_bin_back, max_bin_back) - max_bin_back
            if len(significant_indices) > 0:
                print(
                    f"{LABELS[concept_i]} {incident_i}th incident is very unlikely to happen in segment {[bin_back_segments[ii] for ii in significant_indices]}"
                )


def U_test(activations, free_recall_windows, bin_size=0.25, permutations=5000):
    time_bins = np.arange(0, len(activations) * bin_size, bin_size)
    max_bin_back = 16
    sig_vector_test = []
    bins_back = np.arange(-16, 1)
    activations_width = [4, 6, 8]
    for concept_i, concept_vocalizations in enumerate(free_recall_windows):  # for each concept
        if len(concept_vocalizations) <= 0:
            sig_vector_test.append(np.nan)
            continue

        target_activations = []
        target_activations_indices = []
        for concept_vocalization in concept_vocalizations:  # get the ranges for each concept mention
            # get the average activation score for each aw window prior to bb

            closest_end = np.abs(time_bins - concept_vocalization / 1000).argmin()
            # grab only those bins before concept mention
            temp_tas = []
            if (
                closest_end - (np.max(np.abs(bins_back)) + np.max(activations_width)) >= 0
            ):  # if concept too close to beginning skip it
                for bb in np.abs(bins_back):
                    for aw in activations_width:
                        temp_tas.append(
                            np.mean(
                                activations[
                                    closest_end - (bb + aw) : closest_end - bb,
                                    concept_i,
                                ]
                            )
                        )
                        target_activations_indices.extend(
                            np.arange(closest_end - (bb + aw), closest_end - bb)
                        )  # grab all the indices used in the average not just the bb start
            target_activations.append(temp_tas)
        # Create a mask to exclude the specified concept indices
        target_activations_indices = sorted(set(target_activations_indices))
        mask = np.ones(len(activations), dtype=bool)
        mask[target_activations_indices] = False  # remove these idxs from consideration
        start_indices = np.arange(0, np.max(np.abs(bins_back)) + np.max(activations_width) + 1)
        mask[start_indices] = False

        mask_bins = np.where(mask)[0]
        surrogate_activations = []
        for selected_index in mask_bins:
            temp_surrogate = []
            for bb in np.abs(bins_back):
                for aw in activations_width:
                    if selected_index - (bb + aw) >= 0:
                        temp_act = activations[selected_index - (bb + aw) : selected_index - bb, concept_i]
                        temp_surrogate.append(np.mean(temp_act))
            surrogate_activations.append(temp_surrogate)

        target_activations = np.array(target_activations)
        surrogate_activations = np.array(surrogate_activations)

        for incident_i in range(target_activations.shape[0]):
            t_results = []
            p_results = []
            # for perm in range(permutations):
            #     sample_indices = random.sample(list(np.arange(0, surrogate_activations.shape[0])), target_activations.shape[0])
            #     statistic, p_value = ttest_rel(target_activations.mean(0), surrogate_activations[sample_indices].mean(0))
            #     t_results.append(statistic)
            #     p_results.append(p_value)
            for ii in range(surrogate_activations.shape[0]):
                statistic, p_value = ttest_rel(target_activations[incident_i], surrogate_activations[ii])
                # statistic, p_value = mannwhitneyu(target_activations.mean(0), surrogate_activations[ii])
                if np.isnan(statistic).any():
                    continue
                t_results.append(statistic)
                p_results.append(p_value)

            fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex="col")
            axs[0].plot(t_results)
            axs[0].set_ylabel("t-statistic")
            axs[0].get_yaxis().set_label_coords(-0.08, 0.5)

            axs[1].plot(p_results)
            axs[1].axhline(y=np.mean(p_results), color="red", linestyle="dashed")
            axs[1].set_ylabel("p-value")
            axs[1].get_yaxis().set_label_coords(-0.08, 0.5)
            # axs[1].set_ylim([0, 1])

            plt.suptitle(f"{LABELS[concept_i]} {incident_i}th")
            plt.tight_layout()
            plt.show()
            plt.close()

        # t_results = []
        # p_results = []
        # for perm in range(permutations):
        #     sample_indices = random.sample(list(np.arange(0, surrogate_activations.shape[0])), target_activations.shape[0])
        #     statistic, p_value = ttest_rel(target_activations.mean(0), surrogate_activations[sample_indices].mean(0))
        #     if np.isnan(statistic).any():
        #         continue
        #     t_results.append(statistic)
        #     p_results.append(p_value)
        # sr_stat,sr_p = ttest_1samp(t_results, 0)

        # for ii in range(surrogate_activations.shape[0]):
        #     # statistic, p_value = ttest_rel(target_activations.mean(0), surrogate_activations[ii])
        #     statistic, p_value = mannwhitneyu(target_activations.mean(0), surrogate_activations[ii])
        #     t_results.append(statistic)
        #     p_results.append(p_value)

        # sig_counter = []
        # for perm in range(permutations):
        #     sample_indices = random.sample(list(np.arange(0, surrogate_activations.shape[0])), target_activations.shape[0])
        #     if target_activations.mean() > surrogate_activations[sample_indices].mean():
        #         sig_counter.append(1)
        #     else:
        #         sig_counter.append(0)
        # concept_Ps = 1-sum(sig_counter)/permutations
        # print('done')
