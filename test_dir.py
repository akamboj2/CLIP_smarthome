import os

base = "/home/abhi/research/SmartHome/Data/youhome_mp4_data/mp4data"
one_p = os.listdir(os.path.join(base,'p101'))
labels_text = []
sentances = {"Cook.Cut"             :"cooking by cutting something.",
            "Cook.Usemicrowave"     :"cooking using a microwave",
            "Cook.Useoven"          :"cooking using an oven",
            "Cook.Usestove"         :"cooking using a stove",
            "Drink.Frombottle"      :"drinking from a bottle",
            "Drink.Fromcup"         :"drinking from a cup",
            "Eat.Snack"             :"eating a snack",
            "Eat.Useutensil"        :"eating using a utensil",
            "Exercise"              :"exercising",
            "Getup"                 :"getting up",
            "Lay.Onbed"             :"laying on a bed",
            "Nap"                   :"napping",
            "Play.Boardgame"        :"playing a boardgame",
            "Read"                  :"reading",
            "Use.Coffeemachine"     :"using a coffee machine",
            "Use.Computer"          :"using a computer",
            "Use.Dishwasher"        :"using a dishwasher",
            "Use.Gamecontroller"    :"using a gname controller",
            "Use.Kettle"            :"using a kettle",
            "Use.Mop"               :"using a mop",
            "Use.Phone"             :"using a phone",
            "Use.Refrig"            :"using a refrigerator",
            "Use.Shelf"             :"using a shelf",
            "Use.Sink"              :"using a sink",
            "Use.Switch"            :"using a ninetendo switch",
            "Use.Tablet"            :"using a tablet",
            "Use.Vaccum"            :"using a vaccum",
            "Watch.TV"              :"watching TV",
            "Write"                 :"writing"
            }

for opt in one_p:
    if opt not in sentances:
        print("ERRROR NO", opt)
    
print('finished')
