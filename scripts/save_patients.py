import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from brain_decoding.config.file_path import PATIENTS_FILE_PATH, SURROGATE_FILE_PATH
from brain_decoding.dataloader.patients import Patients


def read_annotation(annotation_file: str) -> List[int]:
    """
    read time stamp (1st column) from annotation files

    :param annotation_file:
    :return:
    """
    annotation_path = Path(__file__).resolve().parents[3] / "data" / "annotations"
    data = pd.read_csv(
        os.path.join(annotation_path, annotation_file),
        sep="^([^\s]*)\s",
        engine="python",
        header=None,
    )
    data[1] = pd.to_numeric(data[1], errors="coerce")
    data.dropna(subset=[1], inplace=True)
    res = np.floor(data[1]).astype(int)
    res = res.tolist()

    return res


if __name__ == "__main__":
    # fmt: off
    annotations = [ "562_FR1", "562_FR2", "563_FR1", "563_FR2", "566_FR1", "566_CR1", "566_FR2", "566_CR2", "567_FR1",
                    "567_CR1", "567_FR2", "567_CR2", "568_FR1", "568_CR1", "572_FR1", "572_CR1", "572_FR2", "572_CR2",
                    "i728_FR1a", "i728_FR1b", "i728_CR1", "i728_FR2", "i728_CR2", ]

    surrogate_windows = Patients()
    for annotation in annotations:
        patient_id, experiment = annotation.split("_")
        patient_id = patient_id.replace("i", "1")
        experiment = experiment.replace("FR", "free_recall")
        experiment = experiment.replace("CR", "cued_recall")
        surrogate_windows.add_experiment(patient_id, experiment)
        surrogate_windows[patient_id][experiment]["annotation"] = read_annotation(f"{annotation}.ann")


    # 2023-06-08 define 2nd value to concept exactly and will use only that one
    patients = Patients()

    # p555
    patients.add_experiment(patient_id="555", experiment_name="free_recall1")
    patients["555"]["free_recall1"]["attacks/bomb/bus/explosion"] = [35914, 93905]
    patients["555"]["free_recall1"]["attacks/bomb/bus/explosion"].description = "including negotiation...more commonly what they said"
    patients["555"]["free_recall1"]["hostage/exchange/sacrifice/negotiations"] = [9563, 22429, 28464, 47244, 54959, 62789, 71765, 101644]
    patients["555"]["free_recall1"]["Jack Bauer"] = [12589, 61419, 106804, 112053]
    patients["555"]["free_recall1"]["Abu Fayed"] = [31634, 45969, 75902, 99888, 122944]
    patients["555"]["free_recall1"]["Abu Fayed"].description = "main terrorist"
    patients["555"]["free_recall1"]["Ahmed Amar"] = [118014]
    patients["555"]["free_recall1"]["Ahmed Amar"].description = "kid"
    patients["555"]["free_recall1"]["President"] = [5725, 51909]

    # p562, exp 5.
    patients.add_experiment(patient_id="562", experiment_name="free_recall1")
    patients["562"]["free_recall1"]["LA"] = [47894]
    patients["562"]["free_recall1"]["attacks/bomb/bus/explosion"] = [16662, 29149, 42753, 79223, 94616, 106387, 150762, 154253, 219886]
    patients["562"]["free_recall1"]["CIA/FBI"] = [9763, 210852, 273087, 300741]
    patients["562"]["free_recall1"]["hostage/exchange/sacrifice"] = [53874, 70537, 124121, 207739, 308822]
    patients["562"]["free_recall1"]["handcuff/chair/tied"] = [259365]
    patients["562"]["free_recall1"]["Jack Bauer"] = [62337, 74338, 137758, 139410, 223847, 248686, 286911, 321887]
    patients["562"]["free_recall1"]["Abu Fayed"] = [133129, 140493, 233770, 263705, 286149, 346257]
    patients["562"]["free_recall1"]["Ahmed Amar"] = [191496, 333707, 337780, 343262, 350226]
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
    patients["563"]["free_recall1"]["hostage/exchange/sacrifice"] = [107116, 139963, 299632]
    patients["563"]["free_recall1"]["hostage/exchange/sacrifice"].description = "hostage/exchange/sacrifice/martyr"
    patients["563"]["free_recall1"]["Jack Bauer"] = [104836, 124938, 134708, 223847, 263527, 278167]
    patients["563"]["free_recall1"]["Chloe"] = [50858]
    patients["563"]["free_recall1"]["Bill"] = [118876]
    patients["563"]["free_recall1"]["Abu Fayed"] = [208077, 228022, 237282, 274762]
    patients["563"]["free_recall1"]["Ahmed Amar"] = [167473, 182602, 194092]
    patients["563"]["free_recall1"]["President"] = [58163, 63165, 91180]

    # p563, Exp 12
    patients.add_experiment(patient_id="563", experiment_name="free_recall2")
    patients["563"]["free_recall2"]["attacks/bomb/bus/explosion"] = [9870, 24960]
    patients["563"]["free_recall2"]["CIA/FBI"] = [57000, 76653, 135626, 165960, 228360]
    patients["563"]["free_recall2"]["hostage/exchange/sacrifice"] = [172402, 183053, 200508, 252889]
    patients["563"]["free_recall2"]["handcuff/chair/tied"] = [218790]
    patients["563"]["free_recall2"]["Jack Bauer"] = [166527, 183570, 197798, 203470, 251550, 262637, 272425, 280108, 290238, 301530]
    patients["563"]["free_recall2"]["Abu Fayed"] = [145580, 221910, 250350, 255895, 263374, 298432]
    patients["563"]["free_recall2"]["Ahmed Amar"] = [63060, 69570]
    patients["563"]["free_recall2"]["President"] = [85319, 128346]

    # p564, exp 3
    patients.add_experiment(patient_id="564", experiment_name="free_recall1")
    patients["564"]["free_recall1"]["LA"] = [110429, 157973]
    patients["564"]["free_recall1"]["attacks/bomb/bus/explosion"] = [12064, 17366, 35598, 108439]
    patients["564"]["free_recall1"]["CIA/FBI"] = [43720, 47634]
    patients["564"]["free_recall1"]["hostage/exchange/sacrifice"] = [97212, 136012, 186583]
    patients["564"]["free_recall1"]["handcuff/chair/tied"] = [154512, 163743]
    patients["564"]["free_recall1"]["Jack Bauer"] = [83685, 128231, 133557, 153427, 170833, 178043, 181438, 189123, 236780, 238545]
    patients["564"]["free_recall1"]["Chloe"] = [55194, 201640, 204830]
    patients["564"]["free_recall1"]["Bill"] = [165258, 169078, 237995]
    patients["564"]["free_recall1"]["Abu Fayed"] = [120224, 124124, 127956, 141783]
    patients["564"]["free_recall1"]["President"] = [65662, 93706, 208315, 213488]

    # p564, Exp 5
    patients.add_experiment(patient_id="564", experiment_name="free_recall2")
    patients["564"]["free_recall2"]["LA"] = [2308]
    patients["564"]["free_recall2"]["attacks/bomb/bus/explosion"] = [8673, 16720, 53178, 110881, 114771]
    patients["564"]["free_recall2"]["white house/DC"] = [83441]
    patients["564"]["free_recall2"]["CIA/FBI"] = [63304, 235019]
    patients["564"]["free_recall2"]["hostage/exchange/sacrifice"] = [223381, 238059]
    patients["564"]["free_recall2"]["handcuff/chair/tied"] = [199782, 231391]
    patients["564"]["free_recall2"]["Jack Bauer"] = [118506, 129104, 139320, 148855, 168405, 198367, 206866, 217996, 289288, 296563]
    patients["564"]["free_recall2"]["Chloe"] = [75761, 234356, 258988]
    patients["564"]["free_recall2"]["Bill"] = [190338, 239774, 250218]
    patients["564"]["free_recall2"]["Abu Fayed"] = [108591, 133145]
    patients["564"]["free_recall2"]["President"] = [82496, 89826, 136825, 216796, 248324]

    # p565, exp 6
    patients.add_experiment(patient_id="565", experiment_name="free_recall1")
    patients["565"]["free_recall1"]["LA"] = [67455, 74175, 108990]
    patients["565"]["free_recall1"]["attacks/bomb/bus/explosion"] = [4519]
    patients["565"]["free_recall1"]["white house/DC"] = [227120]
    patients["565"]["free_recall1"]["CIA/FBI"] = [145655]
    patients["565"]["free_recall1"]["Jack Bauer"] = [24829, 179910, 233590]
    patients["565"]["free_recall1"]["Chloe"] = [223345, 228095]

    # p565, exp 8
    patients.add_experiment(patient_id="565", experiment_name="free_recall2")
    patients["565"]["free_recall2"]["LA"] = [64621, 121530, 159526, 179880, 303209, 329610]
    patients["565"]["free_recall2"]["attacks/bomb/bus/explosion"] = [59250, 116640, 257613]
    patients["565"]["free_recall2"]["white house/DC"] = [252954]
    patients["565"]["free_recall2"]["hostage/exchange/sacrifice"] = [140070, 147540]
    patients["565"]["free_recall2"]["Jack Bauer"] = [23970, 135551]

    # p566, Exp 7 free recall
    patients.add_experiment(patient_id="566", experiment_name="free_recall1")
    patients["566"]["free_recall1"]["attacks/bomb/bus/explosion"] = [19680, 42709]
    patients["566"]["free_recall1"]["hostage/exchange/sacrifice"] = [54737, 89670, 101477, 180725, 279089]
    patients["566"]["free_recall1"]["handcuff/chair/tied"] = [235425]
    patients["566"]["free_recall1"]["Jack Bauer"] = [16510, 55356, 68015, 78712, 100120, 117136, 193986, 206820, 224710, 250371, 263760, 278970]
    patients["566"]["free_recall1"]["Abu Fayed"] = [24190, 174441, 196404, 209220]
    patients["566"]["free_recall1"]["Ahmed Amar"] = [135330, 148170, 157860, 177839, 189184, 214261]
    patients["566"]["free_recall1"]["President"] = [7878, 48987]

    # p566, Exp 7 cued recall
    patients.add_experiment(patient_id="566", experiment_name="cued_recall1")
    patients["566"]["cued_recall1"]["LA"] = [242093]
    patients["566"]["cued_recall1"]["attacks/bomb/bus/explosion"] = [187130, 236610, 271189, 366764]
    patients["566"]["cued_recall1"]["white house/DC"] = [463945]
    patients["566"]["cued_recall1"]["CIA/FBI"] = [40331, 461605, 467597, 503405, 522132]
    patients["566"]["cued_recall1"]["hostage/exchange/sacrifice"] = [36374, 151800, 181740, 228104, 303372, 363247, 417022, 494434]
    patients["566"]["cued_recall1"]["handcuff/chair/tied"] = [345172, 357764, 387810, 396268, 404770, 410571, 428764, 436569]
    patients["566"]["cued_recall1"]["Jack Bauer"] = [4934, 26904, 32845, 53553, 216072, 221191, 302160, 312600, 318530, 344745, 352170, 361366, 375642, 388110, 396090, 406890, 412800, 419056, 426493, 438630, 536346]
    patients["566"]["cued_recall1"]["Chloe"] = [28945, 51983, 60260, 509034]
    patients["566"]["cued_recall1"]["Abu Fayed"] = [92005, 144240, 285912, 291158, 297776, 307333, 317590, 327078, 368423, 390960, 434610, 441689, 490350, 519359, 534002]
    patients["566"]["cued_recall1"]["Abu Fayed"].description = "...calls him 'Indu'"
    patients["566"]["cued_recall1"]["Ahmed Amar"] = [86379, 100395, 106426, 119420, 138696, 156971, 322300, 482898, 489930]
    patients["566"]["cued_recall1"]["President"] = [175356, 194610, 203766, 212220, 222252, 465096]

    # p566, Exp 9 free recall
    patients.add_experiment(patient_id="566", experiment_name="free_recall2")
    patients["566"]["free_recall2"]["LA"] = [89026]
    patients["566"]["free_recall2"]["attacks/bomb/bus/explosion"] = [80971, 146633, 153628, 442781]
    patients["566"]["free_recall2"]["white house/DC"] = [7340]
    patients["566"]["free_recall2"]["CIA/FBI"] = [168772, 175068, 275872, 539737, 545742, 575922, 605874, 615624]
    patients["566"]["free_recall2"]["hostage/exchange/sacrifice"] = [386983, 427140, 450253, 499243, 517650, 680875, 751676]
    patients["566"]["free_recall2"]["handcuff/chair/tied"] = [412550, 660575, 673250, 791073]
    patients["566"]["free_recall2"]["Jack Bauer"] = [36977, 74164, 236531, 387448, 406041, 412430, 418460, 424308, 450797, 460271, 474942, 485645, 499500, 564411, 644249, 666272, 674291, 681096, 687488, 702071, 715010, 780883, 791478, 808130, 819887]
    patients["566"]["free_recall2"]["Jack Bauer"].description = "calls him Kai/Kite"
    patients["566"]["free_recall2"]["Chloe"] = [217637, 223420, 231768, 550686, 618514, 637550]
    patients["566"]["free_recall2"]["Abu Fayed"] = [505471, 511930, 525840, 531443, 572697, 579158, 589007, 599850, 682904, 716014, 722036, 746304, 752562, 770731, 797900, 804242]
    patients["566"]["free_recall2"]["Abu Fayed"].description = "main terrorist ANDU"
    patients["566"]["free_recall2"]["Ahmed Amar"] = [268810, 291966, 346811, 358945, 730820, 736850, 743834, 753931, 760160]
    patients["566"]["free_recall2"]["Ahmed Amar"].description = "kid"
    patients["566"]["free_recall2"]["President"] = [14340, 58170, 64020]

    # p566, Exp 9 cued recall
    patients.add_experiment(patient_id="566", experiment_name="cued_recall2")
    patients["566"]["cued_recall2"]["attacks/bomb/bus/explosion"] = [140430, 177659]
    patients["566"]["cued_recall2"]["white house/DC"] = [314031]
    patients["566"]["cued_recall2"]["CIA/FBI"] = [2495, 72150, 210051, 311773, 327041, 332190]
    patients["566"]["cued_recall2"]["hostage/exchange/sacrifice"] = [147883, 262816, 280328]
    patients["566"]["cued_recall2"]["handcuff/chair/tied"] = [285638]
    patients["566"]["cued_recall2"]["Jack Bauer"] = [14400, 149869, 170773, 170773, 195180, 254640, 261324, 277350, 283989]
    patients["566"]["cued_recall2"]["Chloe"] = [1674, 13139, 237199]
    patients["566"]["cued_recall2"]["Bill"] = [320788]
    patients["566"]["cued_recall2"]["Abu Fayed"] = [45150, 171034, 123396, 171034, 198771, 208429]
    patients["566"]["cued_recall2"]["Abu Fayed"].description = "main terrorist...thinks he's ASAAD"
    patients["566"]["cued_recall2"]["Ahmed Amar"] = [73796]
    patients["566"]["cued_recall2"]["President"] = [120053, 157410]

    # p567, Exp 8 free recall
    patients.add_experiment(patient_id="567", experiment_name="free_recall1")
    patients["567"]["free_recall1"]["LA"] = [15204]
    patients["567"]["free_recall1"]["attacks/bomb/bus/explosion"] = [3546, 9036, 21573]
    patients["567"]["free_recall1"]["hostage/exchange/sacrifice"] = [100980, 246225]
    patients["567"]["free_recall1"]["handcuff/chair/tied"] = [206850]
    patients["567"]["free_recall1"]["Jack Bauer"] = [43170, 45930, 51056, 59820, 76260, 96072, 124230, 133500, 165082, 173280, 199200, 204643, 238524]
    patients["567"]["free_recall1"]["Chloe"] = [75025, 89860, 133740]
    patients["567"]["free_recall1"]["Bill"] = [237480, 243060]
    patients["567"]["free_recall1"]["Abu Fayed"] = [117267, 150146, 167197, 180407]
    patients["567"]["free_recall1"]["Abu Fayed"].description = "main terrorist"
    patients["567"]["free_recall1"]["Ahmed Amar"] = []
    patients["567"]["free_recall1"]["Ahmed Amar"].description = "Note he remembers Ahmed as guy you never see, so was careful to only include Fayed references for the terrorist"
    patients["567"]["free_recall1"]["President"] = [37140, 228269]

    # p567, Exp 8 cued recall
    patients.add_experiment(patient_id="567", experiment_name="cued_recall1")
    patients["567"]["cued_recall1"]["attacks/bomb/bus/explosion"] = [178266]
    patients["567"]["cued_recall1"]["white house/DC"] = [306150]
    patients["567"]["cued_recall1"]["CIA/FBI"] = [7383, 35760, 59684, 181620, 202020, 208020]
    patients["567"]["cued_recall1"]["hostage/exchange/sacrifice"] = [244140]
    patients["567"]["cued_recall1"]["handcuff/chair/tied"] = [232500, 240566]
    patients["567"]["cued_recall1"]["Jack Bauer"] = [2024, 154380, 204450, 227160, 234004, 241216, 249060, 256560, 264550, 302848, 317000]
    patients["567"]["cued_recall1"]["Chloe"] = [1080, 6394]
    patients["567"]["cued_recall1"]["Abu Fayed"] = [170730, 197850]
    patients["567"]["cued_recall1"]["Abu Fayed"].description = "main terrorist"
    patients["567"]["cued_recall1"]["Ahmed Amar"] = [64320, 73230]
    patients["567"]["cued_recall1"]["Ahmed Amar"].description = "kid"
    patients["567"]["cued_recall1"]["President"] = [143403]

    # p567, Exp 10 FR2
    patients.add_experiment(patient_id="567", experiment_name="free_recall2")
    patients["567"]["free_recall2"]["attacks/bomb/bus/explosion"] = [9180, 16085, 256516]
    patients["567"]["free_recall2"]["CIA/FBI"] = [114562, 135120, 202405]
    patients["567"]["free_recall2"]["hostage/exchange/sacrifice"] = [168962]
    patients["567"]["free_recall2"]["handcuff/chair/tied"] = [192730]
    patients["567"]["free_recall2"]["Jack Bauer"] = [80145, 86456, 94230, 102690, 143413, 150460, 160716, 169623, 177263, 187955, 228397, 237058, 273503, 278562]
    patients["567"]["free_recall2"]["Chloe"] = [141188, 147554, 157899, 165785, 199922]
    patients["567"]["free_recall2"]["Abu Fayed"] = [90766, 214470, 241958, 248370, 254186, 282059]
    patients["567"]["free_recall2"]["Abu Fayed"].description = "main terrorist"
    patients["567"]["free_recall2"]["Ahmed Amar"] = [314579, 319720]
    patients["567"]["free_recall2"]["Ahmed Amar"].description = "kid"
    patients["567"]["free_recall2"]["President"] = [44550, 61530, 73992]

    # p567, Exp 10 CR2
    patients.add_experiment(patient_id="567", experiment_name="cued_recall2")
    patients["567"]["cued_recall2"]["white house/DC"] = [277410]
    patients["567"]["cued_recall2"]["CIA/FBI"] = [5346, 276126]
    patients["567"]["cued_recall2"]["hostage/exchange/sacrifice"] = [103560, 136593, 203910]
    patients["567"]["cued_recall2"]["handcuff/chair/tied"] = [209760]
    patients["567"]["cued_recall2"]["Jack Bauer"] = [100290, 137523, 175710, 185042, 191460, 198760, 205860, 214780, 224971, 256675]
    patients["567"]["cued_recall2"]["Abu Fayed"] = [32541, 37708, 119428, 132957, 140594, 153993]
    patients["567"]["cued_recall2"]["Abu Fayed"].description = "main terrorist"
    patients["567"]["cued_recall2"]["Ahmed Amar"] = [8820, 30730, 36937, 143737, 274350]
    patients["567"]["cued_recall2"]["Ahmed Amar"].description = "kid"
    patients["567"]["cued_recall2"]["President"] = [62201, 96914]

    # p568, Exp 5 FR1
    patients.add_experiment(patient_id="568", experiment_name="free_recall1")
    patients["568"]["free_recall1"]["attacks/bomb/bus/explosion"] = [5400, 122910, 130101]
    patients["568"]["free_recall1"]["hostage/exchange/sacrifice"] = [41721]
    patients["568"]["free_recall1"]["Jack Bauer"] = [30690, 39810]

    # p568, Exp 5 CR1
    patients.add_experiment(patient_id="568", experiment_name="cued_recall1")
    patients["568"]["cued_recall1"]["attacks/bomb/bus/explosion"] = [110557]
    patients["568"]["cued_recall1"]["CIA/FBI"] = [15151, 91770, 339916, 347157]
    patients["568"]["cued_recall1"]["hostage/exchange/sacrifice"] = [285881, 330894]
    patients["568"]["cued_recall1"]["handcuff/chair/tied"] = [272209]
    patients["568"]["cued_recall1"]["Jack Bauer"] = [17736, 22762, 233668, 241090, 271695, 277351, 282914, 289741, 329685, 356813]
    patients["568"]["cued_recall1"]["Chloe"] = [13414]
    patients["568"]["cued_recall1"]["Abu Fayed"] = [207806, 213743, 219350, 227707, 240813, 347625]
    patients["568"]["cued_recall1"]["Ahmed Amar"] = [101360, 106616]
    patients["568"]["cued_recall1"]["President"] = [135007]

    # p572, Exp 10 FR1
    patients.add_experiment(patient_id="572", experiment_name="free_recall1")
    patients["572"]["free_recall1"]["attacks/bomb/bus/explosion"] = [21243, 43995]
    patients["572"]["free_recall1"]["CIA/FBI"] = [175981]
    patients["572"]["free_recall1"]["hostage/exchange/sacrifice"] = [102625, 110574, 147493, 197735]
    patients["572"]["free_recall1"]["Jack Bauer"] = [113032, 136548, 141325, 145122, 153029, 223104, 228311, 232767]
    patients["572"]["free_recall1"]["Chloe"] = [196430]
    patients["572"]["free_recall1"]["Abu Fayed"] = [119796, 204677, 219329]
    patients["572"]["free_recall1"]["President"] = [63832]

    # p572, Exp 10 CR1
    patients.add_experiment(patient_id="572", experiment_name="cued_recall1")
    patients["572"]["cued_recall1"]["LA"] = [94610, 136372, 312235]
    patients["572"]["cued_recall1"]["CIA/FBI"] = [78115, 331085]
    patients["572"]["cued_recall1"]["handcuff/chair/tied"] = [243450, 272014]
    patients["572"]["cued_recall1"]["Jack Bauer"] = [21672, 197628, 203850, 243450, 248952, 259173, 265739, 276773, 281485]
    patients["572"]["cued_recall1"]["Chloe"] = [6677]
    patients["572"]["cued_recall1"]["Abu Fayed"] = [156150, 183389, 195611, 208522, 213789]
    patients["572"]["cued_recall1"]["Ahmed Amar"] = [66293]

    # p572, Exp 13 FR2
    patients.add_experiment(patient_id="572", experiment_name="free_recall2")
    patients["572"]["free_recall2"]["LA"] = [28240]
    patients["572"]["free_recall2"]["attacks/bomb/bus/explosion"] = [7470, 27361]
    patients["572"]["free_recall2"]["CIA/FBI"] = [125600, 260670]
    patients["572"]["free_recall2"]["hostage/exchange/sacrifice"] = [156030, 170512]
    patients["572"]["free_recall2"]["handcuff/chair/tied"] = [185512]
    patients["572"]["free_recall2"]["Jack Bauer"] = [141721, 156750, 161760, 167580, 172863, 180618, 185838, 209504, 218484, 230121, 236002]
    patients["572"]["free_recall2"]["Chloe"] = [260227]
    patients["572"]["free_recall2"]["President"] = [41462, 64033]

    # p572, Exp 13 CR2
    patients.add_experiment(patient_id="572", experiment_name="cued_recall2")
    patients["572"]["cued_recall2"]["CIA/FBI"] = [3660, 10770, 16889, 86790]
    patients["572"]["cued_recall2"]["handcuff/chair/tied"] = [254610, 261068, 270540]
    patients["572"]["cued_recall2"]["Jack Bauer"] = [22749, 34980, 203096, 209730, 217067, 248850, 260278, 265710, 270870, 276780]
    patients["572"]["cued_recall2"]["Chloe"] = [595, 7764, 20340, 32974]
    patients["572"]["cued_recall2"]["Abu Fayed"] = [183720, 194366, 211576]
    patients["572"]["cued_recall2"]["Ahmed Amar"] = [78403, 85247]
    patients["572"]["cued_recall2"]["President"] = [120774, 128669, 151560, 160832]

    # i702, Exp 046
    patients.add_experiment(patient_id="1702", experiment_name="free_recall1")
    patients["1702"]["free_recall1"]["attacks/bomb/bus/explosion"] = [19951, 72590]
    patients["1702"]["free_recall1"]["CIA/FBI"] = [58791, 178410]
    patients["1702"]["free_recall1"]["hostage/exchange/sacrifice"] = [136710]
    patients["1702"]["free_recall1"]["Jack Bauer"] = [30120, 41760, 128651, 137040, 146047, 164626]
    patients["1702"]["free_recall1"]["Chloe"] = [170680]
    patients["1702"]["free_recall1"]["Abu Fayed"] = [24466, 40683]

    # i702, Exp 048
    patients.add_experiment(patient_id="1702", experiment_name="free_recall2")
    patients["1702"]["free_recall2"]["CIA/FBI"] = [20180, 31200]
    patients["1702"]["free_recall2"]["hostage/exchange/sacrifice"] = [56567]
    patients["1702"]["free_recall2"]["Jack Bauer"] = [47802, 53350, 65005]
    patients["1702"]["free_recall2"]["Abu Fayed"] = [38562, 51235, 194750]
    patients["1702"]["free_recall2"]["Ahmed Amar"] = [147343, 157864, 169073, 175574, 182430, 188565, 205784]

    # i728, Exp 45
    patients.add_experiment(patient_id="1728", experiment_name="free_recall1a")
    patients["1728"]["free_recall1a"]["LA"] = [124558, 417700]
    patients["1728"]["free_recall1a"]["attacks/bomb/bus/explosion"] = [13424, 24000, 50557, 316980]
    patients["1728"]["free_recall1a"]["CIA/FBI"] = [199230, 208440]
    patients["1728"]["free_recall1a"]["hostage/exchange/sacrifice"] = [80203, 107320]
    patients["1728"]["free_recall1a"]["handcuff/chair/tied"] = [117541, 382823]
    patients["1728"]["free_recall1a"]["Jack Bauer"] = [62170, 84994, 103440, 142080, 150217, 179967, 194678, 217477, 223770, 231774, 248430, 353005, 361650, 370760, 385950, 398146]
    patients["1728"]["free_recall1a"]["Abu Fayed"] = [138692, 163526, 170310, 177390, 182730, 241517, 247070, 326094, 394199, 420926]
    patients["1728"]["free_recall1a"]["Ahmed Amar"] = [286350, 304287, 312510, 320400]

    # i728, Exp 46
    patients.add_experiment(patient_id="1728", experiment_name="free_recall1b")
    patients["1728"]["free_recall1b"]["LA"] = [206067, 322500]
    patients["1728"]["free_recall1b"]["attacks/bomb/bus/explosion"] = [4891, 20927, 36013, 92160, 109980]
    patients["1728"]["free_recall1b"]["CIA/FBI"] = [60630, 240480, 309000, 357990]
    patients["1728"]["free_recall1b"]["hostage/exchange/sacrifice"] = [100460, 156930]
    patients["1728"]["free_recall1b"]["handcuff/chair/tied"] = [209310]
    patients["1728"]["free_recall1b"]["Jack Bauer"] = [73590, 98070, 114090, 119469, 131550, 139241, 159981, 168676, 175678, 183849, 197370, 209640, 215610, 225240, 251970, 264600, 276540, 291390]
    patients["1728"]["free_recall1b"]["Abu Fayed"] = [53663, 82067, 218841, 296044, 306599, 464034, 480602]
    patients["1728"]["free_recall1b"]["Ahmed Amar"] = [350250, 355200, 376331, 385918, 391050, 409250, 426330, 434124, 446656, 460350, 467970, 473436]

    offset_i728 = ((55 * 60 + 45) - (48 * 60 + 36)) * 1000
    patients.add_experiment(patient_id="1728", experiment_name="free_recall1")
    patients["1728"]["free_recall1"].add_events(patients["1728"]["free_recall1a"].events)
    patients["1728"]["free_recall1"].extend_events(patients["1728"]["free_recall1b"].events, offset_i728)

    surrogate_windows.add_experiment(patient_id="1728", experiment_name="free_recall1")
    surrogate_windows["1728"]["free_recall1"].add_events(surrogate_windows["1728"]["free_recall1a"].events)
    surrogate_windows["1728"]["free_recall1"].extend_events(surrogate_windows["1728"]["free_recall1b"].events, offset_i728)

    # i728, Exp 46
    patients.add_experiment(patient_id="1728", experiment_name="cued_recall1")
    patients["1728"]["cued_recall1"]["LA"] = [291690]
    patients["1728"]["cued_recall1"]["attacks/bomb/bus/explosion"] = [64362]
    patients["1728"]["cued_recall1"]["white house/DC"] = [328410, 462510]
    patients["1728"]["cued_recall1"]["CIA/FBI"] = [2280, 321780, 459420, 466108]
    patients["1728"]["cued_recall1"]["hostage/exchange/sacrifice"] = [128250, 253026]
    patients["1728"]["cued_recall1"]["handcuff/chair/tied"] = [239880]
    patients["1728"]["cued_recall1"]["Jack Bauer"] = [8916, 119366, 127470, 136860, 155736, 208135, 239019, 247266, 255986, 267715, 472470]
    patients["1728"]["cued_recall1"]["Chloe"] = [1360, 457724, 471787, 478483]
    patients["1728"]["cued_recall1"]["Abu Fayed"] = [175666, 194935, 210254]
    patients["1728"]["cued_recall1"]["Ahmed Amar"] = [79583]
    patients["1728"]["cued_recall1"]["President"] = [120390, 135540, 154625, 160680]

    # i728, Exp 50
    patients.add_experiment(patient_id="1728", experiment_name="free_recall2")
    patients["1728"]["free_recall2"]["LA"] = [28590, 232280, 440880]
    patients["1728"]["free_recall2"]["attacks/bomb/bus/explosion"] = [4870, 20223, 27840, 66600, 451710]
    patients["1728"]["free_recall2"]["CIA/FBI"] = [272070, 305430, 317880, 475740, 500310]
    patients["1728"]["free_recall2"]["hostage/exchange/sacrifice"] = [60744, 81600, 95564, 124740, 176656]
    patients["1728"]["free_recall2"]["handcuff/chair/tied"] = [239640, 419790]
    patients["1728"]["free_recall2"]["Jack Bauer"] = [63107, 73980, 79746, 89137, 96457, 104100, 124140, 132734, 177145, 187080, 193680, 203183, 218190, 240330, 249254, 273420, 282690, 313290, 327450, 346590, 353250, 363129, 374280, 383730, 391410, 405750, 420660, 609310]
    patients["1728"]["free_recall2"]["Chloe"] = [259081]
    patients["1728"]["free_recall2"]["Abu Fayed"] = [53268, 155534, 292560, 340740, 351690, 361530, 603336, 613840]
    patients["1728"]["free_recall2"]["Ahmed Amar"] = [474450, 501990, 519120, 533370, 549010, 560693, 575220, 586431, 594630, 618970]
    patients["1728"]["free_recall2"]["President"] = [114000, 186030, 257460]

    # i728, Exp 50
    patients.add_experiment(patient_id="1728", experiment_name="cued_recall2")
    patients["1728"]["cued_recall2"]["LA"] = [310560]
    patients["1728"]["cued_recall2"]["attacks/bomb/bus/explosion"] = [156120, 197400, 305109]
    patients["1728"]["cued_recall2"]["white house/DC"] = [322500]
    patients["1728"]["cued_recall2"]["CIA/FBI"] = [73678, 325539]
    patients["1728"]["cued_recall2"]["Jack Bauer"] = [10860, 18193, 125261, 131070, 144810, 157410, 164670, 187358, 203357, 213960, 240885, 251820, 258570, 275777]
    patients["1728"]["cued_recall2"]["Chloe"] = [1771]
    patients["1728"]["cued_recall2"]["Abu Fayed"] = [181358, 199348, 205560, 211024]
    patients["1728"]["cued_recall2"]["Ahmed Amar"] = [70110, 78447]
    patients["1728"]["cued_recall2"]["President"] = [126630, 133200, 141120, 146940, 153240]

    # fmt: on

    # save patients to json file for each patient separately:
    patients.export_json(PATIENTS_FILE_PATH)
    surrogate_windows.export_json(SURROGATE_FILE_PATH)
