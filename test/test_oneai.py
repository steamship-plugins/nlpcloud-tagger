import logging

from src.oneai import *

__copyright__ = "Steamship"
__license__ = "MIT"

from test.util import get_oneai_client, read_test_file

CALL_CENTER_SKILL_EMOTION_ENTITY = {
    'input_text': "Damon:\nhello\n\nDamon:\nIs there somewhere that describes what the beeps mean? It’s so loud it hurts my ears!\n\nClara:\nHello Damon, welcome to the Chatcaffe Support Service. I am Clara here to help you with your issue.\n\nDamon:\nHi\n\nClara:\nI'm sorry you feel that way, it's embarrassing.\n\nClara:\nIn order to serve you more effectively, can you please explain in brief the issue faced by you?\n\nDamon:\nsure\n\nDamon:\nI power on the system, and immediately I get these annoying beeping noises!\n\nClara:\nDamon, can you please let me know the exact number of beeps you get?\n\nDamon:\ntotal? About 8\n\nDamon:\njust like those above\n\nClara:\nJust to confirm, are you getting one long and three short , three long and one short beep after you start your system?\n\nDamon:\nall are short.\n\nClara:\nWhen did the system start doing this? Also, it would be very helpful to know whether you have recently installed any new hardware or software?\n\nDamon:\nfrom what I understand there was a light pen interface card installed on this machine some time ago, it is removed and no longer available. I do not know when it last worked properly.\n\nDamon:\nthere is no other hardware on the machine, I also took the liberty of disconnecting the floppy, cdrom, and HD, to see if that changed anything, it didn't\n\nDamon:\ntried different RAM as well.\n\nClara:\nPlease hold on for a moment, while I search for a resolution for your issue.\n\nDamon:\nsure\n\nClara:\nHave you tried reseating the RAM in the PCI slot?\n\nDamon:\nthere is nothing in the pci slots.\n\nDamon:\nI have tried different ram though\n\nClara:\nThe issue you are facing is due to not installing the memory in the PCI slots and hence you are getting the beeps. I suggest you to please insert the memory in the PCI slots and then restart the system. Please take your time and try out the solution. If you have any problems, please feel free to contact us again. We are here 24 hours a day, 7days a week to assist you.\n\nClara:\nIs that alright with you?\n\nDamon:\nAre you kidding me? RAM can't fit into a PCI slot it doesn't physically go in\n\nDamon:\nSDRAM can only fit into DIMM sockets\n\nClara:\nAre you using the memory cards which were orginally shipped along with the system?\n\nDamon:\nyes\n\nDamon:\nand tried new ones\n\nClara:\nDamon:, to resolve the issue further, I suggest you to please try to install the earlier SDRAM memory cards in the slots and then boot the system. I am sure this will help to fix the issue faced by you.\n\nDamon:\nalready did. the ram must be bad then\n\nDamon:\nis there replacement ram available?\n\nDamon:\nis it just normal sdram? looks like it.\n\nClara:\nDo you wish to have information on SDRAM installed on this system?\n\nDamon:\nsure\n\nClara:\nI am sending you the required web page which contains the details.\n\nClara:\nhttp://www.Chatcaffe.com/longmeaninglessurl.html\n\nClara:\nDid you get the web page?\n\nDamon:\nyes, is there a limit to the maximum amount of ram on the memory module?\n\nClara:\nI am sending you a web page that will provide you details regarding this, please review the web page.\n\nDamon:\noh, and if I needed to find out what various post error beeps mean, what is the name of the BIOS used on the system?\n\nClara:\nhttp://www.ChatCaffe.com/2ndlongmeaninglessurl.shtml\n\nDamon:\ngot it.\n\nClara:\nThe motherboard has Phoenix bios.\n\nClara:\nIs that fine with you?\n\nDamon:\nyep,\n\nClara:\nThank you Damon, for your valuable time. It was a pleasure to assist you.\n\nDamon:\nThanks Clara\n",
    'status': 'success', 'output': [{'text_generated_by_step_name': 'input', 'text_generated_by_step_id': 0,
                                     'text': "Damon:\nhello\n\nDamon:\nIs there somewhere that describes what the beeps mean? It’s so loud it hurts my ears!\n\nClara:\nHello Damon, welcome to the Chatcaffe Support Service. I am Clara here to help you with your issue.\n\nDamon:\nHi\n\nClara:\nI'm sorry you feel that way, it's embarrassing.\n\nClara:\nIn order to serve you more effectively, can you please explain in brief the issue faced by you?\n\nDamon:\nsure\n\nDamon:\nI power on the system, and immediately I get these annoying beeping noises!\n\nClara:\nDamon, can you please let me know the exact number of beeps you get?\n\nDamon:\ntotal? About 8\n\nDamon:\njust like those above\n\nClara:\nJust to confirm, are you getting one long and three short , three long and one short beep after you start your system?\n\nDamon:\nall are short.\n\nClara:\nWhen did the system start doing this? Also, it would be very helpful to know whether you have recently installed any new hardware or software?\n\nDamon:\nfrom what I understand there was a light pen interface card installed on this machine some time ago, it is removed and no longer available. I do not know when it last worked properly.\n\nDamon:\nthere is no other hardware on the machine, I also took the liberty of disconnecting the floppy, cdrom, and HD, to see if that changed anything, it didn't\n\nDamon:\ntried different RAM as well.\n\nClara:\nPlease hold on for a moment, while I search for a resolution for your issue.\n\nDamon:\nsure\n\nClara:\nHave you tried reseating the RAM in the PCI slot?\n\nDamon:\nthere is nothing in the pci slots.\n\nDamon:\nI have tried different ram though\n\nClara:\nThe issue you are facing is due to not installing the memory in the PCI slots and hence you are getting the beeps. I suggest you to please insert the memory in the PCI slots and then restart the system. Please take your time and try out the solution. If you have any problems, please feel free to contact us again. We are here 24 hours a day, 7days a week to assist you.\n\nClara:\nIs that alright with you?\n\nDamon:\nAre you kidding me? RAM can't fit into a PCI slot it doesn't physically go in\n\nDamon:\nSDRAM can only fit into DIMM sockets\n\nClara:\nAre you using the memory cards which were orginally shipped along with the system?\n\nDamon:\nyes\n\nDamon:\nand tried new ones\n\nClara:\nDamon:, to resolve the issue further, I suggest you to please try to install the earlier SDRAM memory cards in the slots and then boot the system. I am sure this will help to fix the issue faced by you.\n\nDamon:\nalready did. the ram must be bad then\n\nDamon:\nis there replacement ram available?\n\nDamon:\nis it just normal sdram? looks like it.\n\nClara:\nDo you wish to have information on SDRAM installed on this system?\n\nDamon:\nsure\n\nClara:\nI am sending you the required web page which contains the details.\n\nClara:\nhttp://www.Chatcaffe.com/longmeaninglessurl.html\n\nClara:\nDid you get the web page?\n\nDamon:\nyes, is there a limit to the maximum amount of ram on the memory module?\n\nClara:\nI am sending you a web page that will provide you details regarding this, please review the web page.\n\nDamon:\noh, and if I needed to find out what various post error beeps mean, what is the name of the BIOS used on the system?\n\nClara:\nhttp://www.ChatCaffe.com/2ndlongmeaninglessurl.shtml\n\nDamon:\ngot it.\n\nClara:\nThe motherboard has Phoenix bios.\n\nClara:\nIs that fine with you?\n\nDamon:\nyep,\n\nClara:\nThank you Damon, for your valuable time. It was a pleasure to assist you.\n\nDamon:\nThanks Clara\n",
                                     'labels': [
                                         {'type': 'emotion', 'skill': 'emotion', 'name': 'anger', 'span': [76, 106],
                                          'value': None, 'output_spans': [{'section': 1, 'start': 55, 'end': 85}],
                                          'input_spans': None, 'span_text': 'It’s so loud it hurts my ears!',
                                          'data': None},
                                         {'type': 'emotion', 'skill': 'emotion', 'name': 'sadness', 'span': [234, 281],
                                          'value': None, 'output_spans': [{'section': 4, 'start': 0, 'end': 47}],
                                          'input_spans': None,
                                          'span_text': "I'm sorry you feel that way, it's embarrassing.", 'data': None},
                                         {'type': 'emotion', 'skill': 'emotion', 'name': 'anger', 'span': [407, 482],
                                          'value': None, 'output_spans': [{'section': 7, 'start': 0, 'end': 75}],
                                          'input_spans': None,
                                          'span_text': 'I power on the system, and immediately I get these annoying beeping noises!',
                                          'data': None}, {'type': 'emotion', 'skill': 'emotion', 'name': 'surprise',
                                                          'span': [1967, 1986], 'value': None,
                                                          'output_spans': [{'section': 24, 'start': 0, 'end': 19}],
                                                          'input_spans': None, 'span_text': 'Are you kidding me?',
                                                          'data': None},
                                         {'type': 'emotion', 'skill': 'emotion', 'name': 'happiness',
                                          'span': [3310, 3350], 'value': None,
                                          'output_spans': [{'section': 46, 'start': 0, 'end': 40}], 'input_spans': None,
                                          'span_text': 'Thank you Damon, for your valuable time.', 'data': None},
                                         {'type': 'emotion', 'skill': 'emotion', 'name': 'happiness',
                                          'span': [3351, 3383], 'value': None,
                                          'output_spans': [{'section': 46, 'start': 41, 'end': 73}],
                                          'input_spans': None, 'span_text': 'It was a pleasure to assist you.',
                                          'data': None},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'PERSON', 'span': [121, 126],
                                          'value': 'Damon', 'output_spans': [{'section': 2, 'start': 6, 'end': 11}],
                                          'input_spans': None, 'span_text': 'Damon', 'data': None},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'ORG', 'span': [139, 168],
                                          'value': 'the Chatcaffe Support Service',
                                          'output_spans': [{'section': 2, 'start': 24, 'end': 53}], 'input_spans': None,
                                          'span_text': 'the Chatcaffe Support Service', 'data': None},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'PERSON', 'span': [175, 180],
                                          'value': 'Clara', 'output_spans': [{'section': 2, 'start': 60, 'end': 65}],
                                          'input_spans': None, 'span_text': 'Clara', 'data': None},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'PERSON', 'span': [491, 496],
                                          'value': 'Damon', 'output_spans': [{'section': 8, 'start': 0, 'end': 5}],
                                          'input_spans': None, 'span_text': 'Damon', 'data': None},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'CARDINAL', 'span': [575, 582],
                                          'value': 'About 8', 'output_spans': [{'section': 9, 'start': 7, 'end': 14}],
                                          'input_spans': None, 'span_text': 'About 8', 'data': {'value': 8.0}},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'CARDINAL', 'span': [654, 657],
                                          'value': 'one', 'output_spans': [{'section': 11, 'start': 33, 'end': 36}],
                                          'input_spans': None, 'span_text': 'one', 'data': {'value': 1.0}},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'CARDINAL', 'span': [667, 672],
                                          'value': 'three', 'output_spans': [{'section': 11, 'start': 46, 'end': 51}],
                                          'input_spans': None, 'span_text': 'three', 'data': {'value': 3.0}},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'CARDINAL', 'span': [681, 686],
                                          'value': 'three', 'output_spans': [{'section': 11, 'start': 60, 'end': 65}],
                                          'input_spans': None, 'span_text': 'three', 'data': {'value': 3.0}},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'CARDINAL', 'span': [696, 699],
                                          'value': 'one', 'output_spans': [{'section': 11, 'start': 75, 'end': 78}],
                                          'input_spans': None, 'span_text': 'one', 'data': {'value': 1.0}},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'TIME', 'span': [1332, 1340],
                                          'value': 'a moment',
                                          'output_spans': [{'section': 17, 'start': 19, 'end': 27}],
                                          'input_spans': None, 'span_text': 'a moment', 'data': None},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'TIME', 'span': [1881, 1889],
                                          'value': '24 hours',
                                          'output_spans': [{'section': 22, 'start': 327, 'end': 335}],
                                          'input_spans': None, 'span_text': '24 hours', 'data': None},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'DATE', 'span': [1897, 1902],
                                          'value': '7days', 'output_spans': [{'section': 22, 'start': 343, 'end': 348}],
                                          'input_spans': None, 'span_text': '7days', 'data': None},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'PERSON', 'span': [2228, 2233],
                                          'value': 'Damon', 'output_spans': [{'section': 29, 'start': 0, 'end': 5}],
                                          'input_spans': None, 'span_text': 'Damon', 'data': None},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'PERSON', 'span': [3320, 3325],
                                          'value': 'Damon', 'output_spans': [{'section': 46, 'start': 10, 'end': 15}],
                                          'input_spans': None, 'span_text': 'Damon', 'data': None},
                                         {'type': 'entity', 'skill': 'entity', 'name': 'PERSON', 'span': [3399, 3404],
                                          'value': 'Clara', 'output_spans': [{'section': 47, 'start': 7, 'end': 12}],
                                          'input_spans': None, 'span_text': 'Clara', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'beeps', 'span': [64, 69],
                                          'value': 0.046, 'output_spans': [{'section': 1, 'start': 43, 'end': 48}],
                                          'input_spans': None, 'span_text': 'beeps', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'issue', 'span': [208, 213],
                                          'value': 0.066, 'output_spans': [{'section': 2, 'start': 93, 'end': 98}],
                                          'input_spans': None, 'span_text': 'issue', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'damon', 'span': [121, 126],
                                          'value': 0.035, 'output_spans': [{'section': 2, 'start': 6, 'end': 11}],
                                          'input_spans': None, 'span_text': 'Damon', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'issue', 'span': [366, 371],
                                          'value': 0.066, 'output_spans': [{'section': 5, 'start': 76, 'end': 81}],
                                          'input_spans': None, 'span_text': 'issue', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'system', 'span': [422, 428],
                                          'value': 0.057, 'output_spans': [{'section': 7, 'start': 15, 'end': 21}],
                                          'input_spans': None, 'span_text': 'system', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'beeps', 'span': [545, 550],
                                          'value': 0.046, 'output_spans': [{'section': 8, 'start': 54, 'end': 59}],
                                          'input_spans': None, 'span_text': 'beeps', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'damon', 'span': [491, 496],
                                          'value': 0.035, 'output_spans': [{'section': 8, 'start': 0, 'end': 5}],
                                          'input_spans': None, 'span_text': 'Damon', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'system', 'span': [732, 738],
                                          'value': 0.057, 'output_spans': [{'section': 11, 'start': 111, 'end': 117}],
                                          'input_spans': None, 'span_text': 'system', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'system', 'span': [784, 790],
                                          'value': 0.057, 'output_spans': [{'section': 13, 'start': 13, 'end': 19}],
                                          'input_spans': None, 'span_text': 'system', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'ram', 'span': [1292, 1295],
                                          'value': 0.062, 'output_spans': [{'section': 16, 'start': 16, 'end': 19}],
                                          'input_spans': None, 'span_text': 'RAM', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'different ram',
                                          'span': [1282, 1295], 'value': 0.06,
                                          'output_spans': [{'section': 16, 'start': 6, 'end': 19}], 'input_spans': None,
                                          'span_text': 'different RAM', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'issue', 'span': [1383, 1388],
                                          'value': 0.066, 'output_spans': [{'section': 17, 'start': 70, 'end': 75}],
                                          'input_spans': None, 'span_text': 'issue', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'ram', 'span': [1440, 1443],
                                          'value': 0.062, 'output_spans': [{'section': 19, 'start': 29, 'end': 32}],
                                          'input_spans': None, 'span_text': 'RAM', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'pci slots',
                                          'span': [1493, 1502], 'value': 0.1,
                                          'output_spans': [{'section': 20, 'start': 24, 'end': 33}],
                                          'input_spans': None, 'span_text': 'pci slots', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'ram', 'span': [1535, 1538],
                                          'value': 0.062, 'output_spans': [{'section': 21, 'start': 23, 'end': 26}],
                                          'input_spans': None, 'span_text': 'ram', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'different ram',
                                          'span': [1525, 1538], 'value': 0.06,
                                          'output_spans': [{'section': 21, 'start': 13, 'end': 26}],
                                          'input_spans': None, 'span_text': 'different ram', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'pci slots',
                                          'span': [1622, 1631], 'value': 0.1,
                                          'output_spans': [{'section': 22, 'start': 68, 'end': 77}],
                                          'input_spans': None, 'span_text': 'PCI slots', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'pci slots',
                                          'span': [1718, 1727], 'value': 0.1,
                                          'output_spans': [{'section': 22, 'start': 164, 'end': 173}],
                                          'input_spans': None, 'span_text': 'PCI slots', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'issue', 'span': [1558, 1563],
                                          'value': 0.066, 'output_spans': [{'section': 22, 'start': 4, 'end': 9}],
                                          'input_spans': None, 'span_text': 'issue', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'system', 'span': [1749, 1755],
                                          'value': 0.057, 'output_spans': [{'section': 22, 'start': 195, 'end': 201}],
                                          'input_spans': None, 'span_text': 'system', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'beeps', 'span': [1662, 1667],
                                          'value': 0.046, 'output_spans': [{'section': 22, 'start': 108, 'end': 113}],
                                          'input_spans': None, 'span_text': 'beeps', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'memory', 'span': [1608, 1614],
                                          'value': 0.028, 'output_spans': [{'section': 22, 'start': 54, 'end': 60}],
                                          'input_spans': None, 'span_text': 'memory', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'memory', 'span': [1704, 1710],
                                          'value': 0.028, 'output_spans': [{'section': 22, 'start': 150, 'end': 156}],
                                          'input_spans': None, 'span_text': 'memory', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'ram', 'span': [1987, 1990],
                                          'value': 0.062, 'output_spans': [{'section': 24, 'start': 20, 'end': 23}],
                                          'input_spans': None, 'span_text': 'RAM', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'system', 'span': [2173, 2179],
                                          'value': 0.057, 'output_spans': [{'section': 26, 'start': 75, 'end': 81}],
                                          'input_spans': None, 'span_text': 'system', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'memory', 'span': [2116, 2122],
                                          'value': 0.028, 'output_spans': [{'section': 26, 'start': 18, 'end': 24}],
                                          'input_spans': None, 'span_text': 'memory', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'issue', 'span': [2251, 2256],
                                          'value': 0.066, 'output_spans': [{'section': 29, 'start': 23, 'end': 28}],
                                          'input_spans': None, 'span_text': 'issue', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'issue', 'span': [2411, 2416],
                                          'value': 0.066, 'output_spans': [{'section': 29, 'start': 183, 'end': 188}],
                                          'input_spans': None, 'span_text': 'issue', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'system', 'span': [2367, 2373],
                                          'value': 0.057, 'output_spans': [{'section': 29, 'start': 139, 'end': 145}],
                                          'input_spans': None, 'span_text': 'system', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'damon', 'span': [2228, 2233],
                                          'value': 0.035, 'output_spans': [{'section': 29, 'start': 0, 'end': 5}],
                                          'input_spans': None, 'span_text': 'Damon', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'memory', 'span': [2323, 2329],
                                          'value': 0.028, 'output_spans': [{'section': 29, 'start': 95, 'end': 101}],
                                          'input_spans': None, 'span_text': 'memory', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'ram', 'span': [2456, 2459],
                                          'value': 0.062, 'output_spans': [{'section': 30, 'start': 17, 'end': 20}],
                                          'input_spans': None, 'span_text': 'ram', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'ram', 'span': [2506, 2509],
                                          'value': 0.062, 'output_spans': [{'section': 31, 'start': 21, 'end': 24}],
                                          'input_spans': None, 'span_text': 'ram', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'system', 'span': [2636, 2642],
                                          'value': 0.057, 'output_spans': [{'section': 33, 'start': 59, 'end': 65}],
                                          'input_spans': None, 'span_text': 'system', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'web page',
                                          'span': [2695, 2703], 'value': 0.046,
                                          'output_spans': [{'section': 35, 'start': 30, 'end': 38}],
                                          'input_spans': None, 'span_text': 'web page', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'details',
                                          'span': [2723, 2730], 'value': 0.025,
                                          'output_spans': [{'section': 35, 'start': 58, 'end': 65}],
                                          'input_spans': None, 'span_text': 'details', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'web page',
                                          'span': [2813, 2821], 'value': 0.046,
                                          'output_spans': [{'section': 37, 'start': 16, 'end': 24}],
                                          'input_spans': None, 'span_text': 'web page', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'ram', 'span': [2878, 2881],
                                          'value': 0.062, 'output_spans': [{'section': 38, 'start': 47, 'end': 50}],
                                          'input_spans': None, 'span_text': 'ram', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'memory', 'span': [2889, 2895],
                                          'value': 0.028, 'output_spans': [{'section': 38, 'start': 58, 'end': 64}],
                                          'input_spans': None, 'span_text': 'memory', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'web page',
                                          'span': [2931, 2939], 'value': 0.046,
                                          'output_spans': [{'section': 39, 'start': 19, 'end': 27}],
                                          'input_spans': None, 'span_text': 'web page', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'web page',
                                          'span': [3004, 3012], 'value': 0.046,
                                          'output_spans': [{'section': 39, 'start': 92, 'end': 100}],
                                          'input_spans': None, 'span_text': 'web page', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'details',
                                          'span': [2962, 2969], 'value': 0.025,
                                          'output_spans': [{'section': 39, 'start': 50, 'end': 57}],
                                          'input_spans': None, 'span_text': 'details', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'system', 'span': [3131, 3137],
                                          'value': 0.057, 'output_spans': [{'section': 40, 'start': 109, 'end': 115}],
                                          'input_spans': None, 'span_text': 'system', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'beeps', 'span': [3078, 3083],
                                          'value': 0.046, 'output_spans': [{'section': 40, 'start': 56, 'end': 61}],
                                          'input_spans': None, 'span_text': 'beeps', 'data': None},
                                         {'type': 'keyword', 'skill': 'keyword', 'name': 'damon', 'span': [3320, 3325],
                                          'value': 0.035, 'output_spans': [{'section': 46, 'start': 10, 'end': 15}],
                                          'input_spans': None, 'span_text': 'Damon', 'data': None}]}]}


def _check_response(resp):
    assert read_test_file("call_center.txt") == resp.input_text
    assert resp.output is not None
    assert len(resp.output) == 1
    output_block = resp.output[0]
    assert len(output_block.labels) > 0
    for label in output_block.labels:
        tag = label.to_steamship_tag()
        assert tag.name is not None
        assert tag.kind is not None
        if tag.value is not None:
            assert isinstance(tag.value, dict)
        assert tag.start_idx is not None
        assert tag.end_idx is not None

        # Assert that the labels all perfectly capture text.
        label = resp.input_text[tag.start_idx:tag.end_idx]
        assert label[0] != ' '
        assert label[-1] != ' '

        # This is an indirect way to make sure that OneAI is, like us, start-inclusive and end-exclusive in their
        # indexing scheme (Python slice semantics) -- we test the ACTUAL character underneath the end_idx and verify
        # that it is a whitespace or punctuation character (which, for this test dataset, should hold).
        if tag.end_idx < len(resp.input_text):
            assert resp.input_text[tag.end_idx] in [' ', '\n', ',', '.', ':', '?']



def test_oneai_live():
    """Test OneAI client with a live API key. Intended for local development testing."""
    client = get_oneai_client()
    if client is None:
        logging.error("Unable to create live OneAI client. Skipping, rather than failing, test_oneai_live test.")
        return

    request = OneAIRequest(
        text=read_test_file("call_center.txt"),
        input_type=OneAIInputType.conversation,
        steps=[
            {"skill": "emotions"},
            {"skill": "entities"},
            {"skill": "keywords"},
        ]
    )
    response = client.request(request)
    _check_response(response)


def test_oneai_saved():
    """Performs `test_oneai_live` with saved response data, intended for CI/CD use."""
    response = OneAiResponse.from_dict(CALL_CENTER_SKILL_EMOTION_ENTITY)
    _check_response(response)
