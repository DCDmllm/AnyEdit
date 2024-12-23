import random
import torch
from concept.utils import process_text_multi_turn
import spacy

few_example_dict = {
    "add": [
        ['Beautiful cat with mojito sitting in a cafe on the street.',
         "{'edit': 'add a hat to the cat', 'edited object': 'hat', 'output': 'Beautiful cat wearing a hat with mojito sitting in a cafe on the street.'}"],
        ['Robot playing chess at home.',
         "{'edit': 'add a cheerful smiling face.', 'edited object': 'smiling face', 'output': 'robot playing chess at home with a cheerful smiling face.'}"],
        ['An airplaine is flying in the sky in rainy day.',
         "{'edit': 'add flowers in the windows', 'edited object': 'flowers', 'output': 'An airplaine with flowers in the windows is flying in the sky in rainy day.'}"],
        ['Attic Bedroom With Large Ceilings',
         "{'edit': 'add the room with beautiful chandeliers', 'edited object': 'chandeliers', 'output': 'Attic Bedroom With Beautiful Chandeliers on Large Ceilings'}"],
        ['Wedding rings and yellow flower on a red background',
         "{'edit': 'place a bird by the yellow flower', 'edited object': 'bird', 'output': 'Wedding rings, a bird, and yellow flower on a red background'}"],
        ['A photo of mountains and trees',
         "{'edit': 'place a castle between the trees', 'edited object': 'castle', 'output': 'photo of mountains, trees and castle between the trees'}"],
        ['Tree Near the lake in the morning',
         "{'edit': 'add autumn leaves on top of tree', 'edited object': 'leaves', 'output': 'Tree with autumn leaves on top Near the lake in the morning'}"],
        ['robot and alien sitting on hanging bridge at daytime',
         "{'edit': 'add three books in them.', 'edited object': 'three books', 'output': 'robot and alien holding three books while sitting on hanging bridge at daytime'}"],
        ['Skogafoss waterfall in the south of Iceland',
         "{'edit': 'add a colorful rainbow in the backhground!', 'edited object': 'rainbow', 'output': 'Skogafoss waterfall with a colorful rainbow in the south of Iceland'}"],
        ['A serene beach with clear blue water.',
         "{'edit': 'add a sailboat on the horizon', 'edited object': 'sailboat', 'output': 'A serene beach with clear blue water and a sailboat on the horizon.'}"],
        ['A forest path covered in fallen leaves.',
         "{'edit': 'place a wooden bench along the path', 'edited object': 'wooden bench', 'output': 'A forest path covered in fallen leaves with a wooden bench along the path.'}"],
        ['A snowy mountain peak under a clear sky.',
         "{'edit': 'add a flying eagle above the peak', 'edited object': 'eagle', 'output': 'A snowy mountain peak under a clear sky with a flying eagle above.'}"],
        ['A peaceful meadow with wildflowers.',
         "{'edit': 'add a deer grazing in the meadow', 'edited object': 'deer', 'output': 'A peaceful meadow with wildflowers and a deer grazing.'}"],
        ['A modern living room with a large sofa.',
         "{'edit': 'place a coffee table in front of the sofa', 'edited object': 'coffee table', 'output': 'A modern living room with a large sofa and a coffee table in front.'}"],
        ['A beautiful sunset over the ocean.',
         "{'edit': 'add a lighthouse on the shore', 'edited object': 'lighthouse', 'output': 'A beautiful sunset over the ocean with a lighthouse on the shore.'}"],
        ['A classic car parked on a quiet street.',
         "{'edit': 'place a bicycle leaning against the car', 'edited object': 'bicycle', 'output': 'A classic car parked on a quiet street with a bicycle leaning against it.'}"]
    ],
    "remove": [
        ['Beautiful cat wearing a hat with mojito sitting in a cafe on the street.',
         "{'edit': 'remove the hat of the cat', 'edited object': 'hat', 'output': 'Beautiful cat with mojito sitting in a cafe on the street.'}"],
        ['A cute creature and a dog sit at the beach.',
         "{'edit': 'remove the dog besides the creature', 'edited object': 'dog', 'output': 'A cute creature sits at the beach.'}"],
        ['A picturesque lake with a small boat floating in the center.',
         "{'edit': 'remove the small boat from the lake', 'edited object': 'small boat', 'output': 'A picturesque lake.'}"],
        ['A group of children playing soccer in the field.',
         "{'edit': 'remove soccer', 'edited object': 'soccer', 'output': 'A group of children in the field.'}"],
        ['A serene garden with a stone bench and a rabbit hopping nearby.',
         "{'edit': 'remove the rabbit hopping nearby', 'edited object': 'rabbit', 'output': 'A serene garden with a stone bench.'}"],
        ['Little bunny playing with a kite in the park.',
         "{'edit': 'delete the kite.', 'edited object': 'kite', 'output': 'Little bunny in the park.'}"],
        ['Wedding rings, a bird, and yellow flower on a red background.',
         "{'edit': 'erase the bird by the yellow flower', 'edited object': 'bird', 'output': 'Wedding rings and yellow flower on a red background.'}"],
        ['Skogafoss waterfall with a colorful rainbow in the south of Iceland.',
         "{'edit': 'remove the colorful rainbow in the backhground', 'edited object': 'rainbow', 'output': 'Skogafoss waterfall in the south of Iceland.'}"],
        ['A snowy mountain peak under a clear sky with a flying eagle above.',
        "{'edit': 'remove the flying eagle above the peak', 'edited object': 'eagle', 'output': 'A snowy mountain peak under a clear sky.'}"]
    ],
    "replace": [
        ['A beautiful cat wearing a hat with mojito sitting in a cafe on the street.',
         "{'edit': 'alter the cat to dog', 'edited object': 'cat', 'new object': 'dog', 'output': 'Beautiful dog wearing a hat with mojito sitting in a cafe on the street.'}"],
        ['A cute cat and a dog sit at the beach.',
         "{'edit': 'change the cat to a rabbit', 'edited object': 'cat', 'new object': 'rabbit', 'output': 'A cute rabbit and a dog sit at the beach.'}"],
        ['Little bunny playing with a kite in the park.',
         "{'edit': 'alter the kite to a ball.', 'edited object': 'kite', 'new object': 'ball', 'output': 'Little bunny playing with a ball in the park.'}"],
        ['horse on a red Boat Near Mountains During Golden Hour',
         "{'edit': 'replace the Boat to a car', 'edited object': 'Boat', 'new object': 'car', 'output': 'horse on a red car Near Mountains During Golden Hour'}"],
        ['A parrot sitting on a pirate’s shoulder.',
         "{'edit': 'replace the parrot with a cat', 'edited object': 'parrot', 'new object': 'cat', 'output': 'A cat sitting on a pirate’s shoulder.'}"],
        ['A child flying a kite in the sunny field.',
         "{'edit': 'change the kite to a drone', 'edited object': 'kite', 'new object': 'drone', 'output': 'A child flying a drone in the sunny field.'}"],
        ['A magician pulling a rabbit out of a hat.',
         "{'edit': 'change the rabbit to a dove', 'edited object': 'rabbit', 'new object': 'dove', 'output': 'A magician pulling a dove out of a hat.'}"],
        ['A knight in shining armor riding a dragon.',
         "{'edit': 'replace the dragon with a horse', 'edited object': 'dragon', 'new object': 'horse', 'output': 'A knight in shining armor riding a horse.'}"],
        ['A Polar Bear with rubber gloves pushing shopping carts',
         "{'edit': 'change the Polar Bear to a Monkey', 'edited object': 'Polar Bear', 'new object': 'Monkey', 'output': 'A Monkey with rubber gloves pushing shopping carts.'}"]
    ],
    "color_alter": [
        ['There is a horse on a red boat near mountains during golden hour',
         "{'edit': 'Turn the color of boat to blue', 'edited object': 'Boat', 'output': 'There is a horse on a blue boat near mountains during golden hour'}"],
        ['A yellow cat and a dog sit at the beach.',
         "{'edit': 'change the color of cat to black', 'edited object': 'cat', 'output': 'A black cat and a dog sit at the beach.'}"],
        ['A white rabbit lies on the grass.',
         "{'edit': 'change the color of rabbit to black', 'edited object': 'rabbit', 'output': 'A black rabbit lies on the grass.'}"],
        ['A red sports car was parked on the road in front of a quiet residential area.', "{'edit': 'Alter the color of car to pink', 'edited object': 'car', 'output': 'A pink sports car was parked on the road in front of a quiet residential area.'}"],
        ['A man wears a white shirt and a red tie.',
         "{'edit': 'change the color of tie to blue', 'edited object': 'tie', 'output': 'A man wears a white shirt and a blue tie.'}"],
        ['A boy plays with a yellow ball on the playground.',
         "{'edit': 'change the color of ball to red', 'edited object': 'ball', 'output': 'A boy plays with a red ball on the playground.'}"],
        ['A cyclist in a blue jersey rides down the street.',
         "{'edit': 'change the color of jersey to yellow', 'edited object': 'jersey', 'output': 'A cyclist in a yellow jersey rides down the street.'}"],
        ['A black and white dog runs across the field.',
         "{'edit': 'change the color of dog to brown', 'edited object': 'dog', 'output': 'A brown dog runs across the field.'}"],
        ['A woman with a pink scarf walks through the market.',
         "{'edit': 'change the color of scarf to orange', 'edited object': 'scarf', 'output': 'A woman with an orange scarf walks through the market.'}"],
        ['A man with a red hat sits on a bench in a park.',
         "{'edit': 'change the color of hat to green', 'edited object': 'hat', 'output': 'A man with a green hat sits on a bench in a park.'}"]
    ],
    "material_alter": [
        ['The little boy is sitting on a plastic chair reading in the sun.',
         "{'edit': 'turn the material of chair to wooden', 'edited object': 'chair', 'output': 'The little boy is sitting on a wooden chair reading in the sun.'}"],
        ['The little boy is sitting on a plastic chair reading in the sun.',
         "{'edit': 'turn the chair to wooden', 'edited object': 'chair', 'output': 'The little boy is sitting on a wooden chair reading in the sun.'}"],
        ['A white horse is standing in the jungle.',
         "{'edit': 'change the material of horse to statuary', 'edited object': 'horse', 'output': 'A white statuary horse stands in the jungle.'}"],
        ['a colorful tent is stationed on a bike lane',
         "{'edit': 'alter the tent to be transparent', 'edited object': 'tent', 'output': 'A colorful transparent tent is stationed on a bike lane.'}"],
        ['a surfer rides his bike down the street.',
         "{'edit': 'make the bike to be wooden', 'edited object': 'bike', 'output': 'a surfer rides his wooden bike down the street.'}"],
        ['a kitchen with an island and wood accents.',
         "{'edit': 'alter the island to be made of glass', 'edited object': 'island', 'output': 'a kitchen with a glass island and wood accents.'}"],
        ['A white rabbit lies on the grass.',
         "{'edit': 'make the material of rabbit to paper', 'edited object': 'rabbit', 'output': 'A white paper rabbit lies on the grass.'}"]
    ],
    "texture_alter": [
        ['A red sports car was parked on the road in front of a quiet residential area.',
         "{'edit': 'Alter the texture of car to woven', 'edited object': 'car', 'output': 'A red woven sports car was parked on the road in front of a quiet residential area.'}"],
        ['The little boy is sitting on a plastic chair reading in the sun.',
         "{'edit': 'turn the texture of chair to brushy', 'edited object': 'chair', 'output': 'The little boy is sitting on a brushy chair reading in the sun.'}"],
        ['Two cats sitting at the beach.', "{'edit': 'change the texture of cat to meshed', 'edited object': 'cat', 'output': 'Two meshed cats sitting at the beach.'}"],
        ['The little boy is sitting on a plastic chair reading in the sun.',
         "{'edit': 'turn the texture of chair to brushy', 'edited object': 'chair', 'output': 'The little boy is sitting on a brushy chair reading in the sun.'}"],
        ['A white horse is standing in the jungle.',
         "{'edit': 'change the texture of horse to dotted', 'edited object': 'horse', 'output': 'A white dotted horse stands in the jungle.'}"],
        ['A white rabbit lies on the grass.',
         "{'edit': 'make the texture of rabbit to striped', 'edited object': 'rabbit', 'output': 'A white striped rabbit lies on the grass.'}"]
    ],
    "appearance_alter": [
        ['a cake is placed on top of a table.',
         "{'edit': 'make the cake decorated with candles.', 'edited object': 'cake', 'output': 'a cake decorated by candles is placed on top of a table.'}"],
        ['A large tree stands in the middle of the park.',
         "{'edit': 'make the tree full of cherries', 'edited object': 'tree', 'output': 'A large cherry blossom tree stands in the middle of the park.'}"],
        ['a woman is riding a bicycle on a sunny day.',
         "{'edit': 'make the basket full of flowers', 'edited object': 'bicycle', 'output': 'a woman is riding a bicycle with a basket of flowers on a sunny day.'}"],
        ['a tall building is under construction.',
         "{'edit': 'make the building covered in green scaffolding', 'edited object': 'building', 'output': 'a tall building covered in green scaffolding is under construction.'}"],
        ['A young man is walking down the street.',
         "{'edit': 'make the man wear a pair of sunglasses', 'edited object': 'man', 'output': 'A young man with a pair of sunglasses is walking down the street.'}"]
    ],
    "action_change": [
        ['A horse is walking near mountains during golden hour.',
         "{'edit': 'change the action of horse to running', 'edit object': 'horse', 'output': 'A horse is running near mountains during golden hour.'}"],
        ['A black cat is lying on a bed.',
         "{'edit': 'make the action of cat to sitting', 'edit object': 'cat', 'output': 'a black cat is sitting on a bed.'}"],
        ['There is a dog lies on a grass.',
         "{'edit': 'make the action of the dog to eating', 'edit object': 'dog', 'output': 'There is a dog eatting something on a grass.'}"],
        ['A chef is chopping vegetables in the kitchen.',
         "{'edit': 'change the action of the chef to cooking', 'edit object': 'chef', 'output': 'A chef is cooking in the kitchen.'}"],
        ['A baseball player walking across a base on a field.',
         "{'edit': 'turn the action of the player to jumping', 'edit object': 'player', 'output': 'A baseball player running across a base on a field.'}"]
    ],
    "background_change": [
        ['horse on a red boat near mountains during golden hour.',
         "{'edit': 'change the background to a beach', 'new background': 'beach', 'output': 'horse on a red boat near a beach during golden hour.'}"],
        ['A dirty rabbit lies on the grass.',
         "{'edit': 'alter the background to a desert', 'new background': 'desert', 'output': 'A dirty rabbit lies on the desert.'}"],
        ['scooter and feet in sandals on a sidewalk.',
         "{'edit': 'turn the background to a grass', 'new background': 'grass', 'output': 'scooter and feet in sandals on a grass.'}"],
        ['a group of people on skis on snowy slope.',
         "{'edit': 'change the background to a hillside', 'new background': 'hillside', 'output': 'a group of people on skis on hillside.'}"],
        ['three people posing for a picture together while skiing on a snowy mountain.',
         "{'edit': 'change the background to a green mountain', 'new background': 'green mountain', 'output': 'three people posing for a picture together while skiing on a green mountain.'}"]
    ],
    "tone_transfer": [
        ['The little boy is sitting on a green chair reading in the sun.',
         "{'edit': 'Turn the weather to rain', 'new state': 'rain', 'output': 'The little boy is sitting on a green chair reading in the rain.'}"],
        ['A horse is walking near mountains during spring.',
         "{'edit': 'change the season to winter', 'new state': 'winter', 'output': 'A horse is walking near mountains during winter.'}"],
        ['A white dog is lying on a grass in a bright morning.',
         "{'edit': 'make the time to midnight', 'new state': 'midnight', 'output': 'A white dog is lying on a grass in a dark midnight.'}"],
        ['A sports car was parked on the road in a sunny day.',
         "{'edit': 'change the weather to rain and thundery', 'new state': 'rain and thundery', 'output': 'A sports car was parked on the road in a rain and thundery day.'}"],
        ['A busy cafe serves coffee on a bustling morning.',
         "{'edit': 'change the time to evening', 'new state': 'evening', 'output': 'A busy cafe serves coffee on a bustling evening.'}"],
        ['A desk stands in an office in a modern city',
         "{'edit': 'change the time to industrial-age', 'new state': 'industrial-age', 'output': 'A desk stands in an office in an industrial-age city.'}"]
    ],
    "textual": [
        ["'description': 'an old advertisement', 'text': 'sweet coconut'.",
        """{'edit': "Change the text 'sweet coconut' to 'nice coconut'", 'input': "an old advertisement with text 'sweet coconut'.", 'output': "an old advertisement with text 'nice coconut'."}"""],
        ["'description': 'A cat holding a sign', 'text': 'Oh good'.",
        """{'edit': "Change the text 'Oh good' to 'Hello world'", 'input': "A cat holding a sign that says 'Oh good'.", 'output': "A cat holding a sign that says 'Hello world'."}"""],
        ["'description': 'A raccoon stands in front of the blackboard with the words', 'text': 'Deep Learning'.",
         """{'edit': "Change the text 'Deep Learning' to 'Machine Learning'", 'input': "A raccoon stands in front of the blackboard with the words 'Deep Learning' written on it.", 'output': "A raccoon stands in front of the blackboard with the words ‘Machine Learning’ written on it."}"""],
        ["'description': 'A boy with the words on the shirt', 'text': 'lucky'.",
         """{'edit': "Change the text 'lucky' to 'happy'", 'input': "A boy with the words 'lucky' written on the shirt.", 'output': "A boy with the words 'happy' written on the shirt."}"""],
        ["'description': 'A photograph of the colorful graffiti art on the wall with the words', 'text': 'Get Ready to Party'.",
         """{'edit': "Change the text 'Get Ready to Party' to 'Where is the Party'", 'input': "A photograph of the colorful graffiti art on the wall with the words ‘Get Ready to Party’.", 'output': "A photograph of the colorful graffiti art on the wall with the words ‘Where is the Party’."}"""]
    ]
}

def get_content_instruction(new_prompt, few_shot_examples, instruction_type, tokenizer):
    system_message = f"You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is JSON with the following string fields: 'edit', 'edited object','output'. Here is the latest conversation between Assistant and User."
    if instruction_type == 'add':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for adding an object to the image ('edit'), the object to add ('edited object'), and the caption with the object ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {{'edit': '<instruction>', 'edited object': '<object>', 'output': '<caption>'}}. Construct the instruction with one of the following instruction words: ['place', 'add', 'include']. Don't include any \" or edit any actions in the instruction.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'remove':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for removing an object to the image ('edit'), the object to remove ('edited object'), and the caption with the object ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<object>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['erase', 'remove', 'delete']. Don't include any \" or edit any actions in the instruction.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'replace':
        system_message = "You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is JSON with the following string fields: 'edit', 'edited object', 'new object', 'output'. Notice that the 'edited object' should not be human and nouns about human. Here is the latest conversation between Assistant and User."
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for replacing an object to another object in the image ('edit'), the original object to replace ('edited object'), the new object ('new object'), the caption with the new object ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<object>', 'new object': '<new object>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['alter', 'change', 'replace']. Don't include any \" in the instruction and don't use remove instruction. The new object cannot be empty.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, new object, and output caption for you. Let's get started!"]
    elif instruction_type == 'color_alter':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for changing the color of one object in the image ('edit'), the object to change color ('edited object'), the caption of the object with new color ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<original object>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['alter', 'change', 'turn']. Use the following format to construct instruction: {change/alter/turn the color of the <object> to <color>}. Don't include any \" in the response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'material_alter':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for changing an object's material in the image ('edit'), the object to change appearance ('edited object'), the caption with the object of the new appearance ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<original object>', 'output': '<caption>'}. The material should be selected from 'wooden', 'vitreous', 'metallic', 'statuary' and 'paper'. Use the following format to construct instruction: {change/alter/turn/make the material of the <object> to <material>}. Don't include any \" in the instruction and response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'texture_alter':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for changing an object's texture in the image ('edit'), the object to change appearance ('edited object'), the caption with the object of the new appearance ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<original object>', 'output': '<caption>'}. Construct the instruction with one of the following texture words: ['dotted', 'striped', 'brushy', 'woven', 'meshed']. Use the following format to construct instruction: {change/alter/turn/make the texture of the <object> to <texture>}. Don't include any \" in the instruction and response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'appearance_alter':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for changing an object's appearance in the image, such as texture or material ('edit'), the object to change appearance ('edited object'), the caption with the object of the new appearance ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<original object>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['turn', 'make']. Don't include any \" in the instruction and response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'action_change':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for changing an object's action in the image ('edit'), the object to change action ('edited object'), the caption of object with a new action ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<original object>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['change', 'turn', 'make']. Use the following format to construct instruction: {change/turn/make the action of the <object> to <action>}. Don't include any \" in the instruction and only change the action.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'background_change':
        # introduction message #
        system_message = "You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is JSON with the following string fields: 'edit', 'new background', 'output'. Here is the latest conversation between Assistant and User."
        intro_message = [
            "Hi, My job to take a given caption ('input') and to output the following: an instruction for changing the background in the image ('edit'), the new background ('new background'), the caption with the new background ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'new background': '<background>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['alter', 'change', 'turn']. Use the following format to construct instruction: {change/alter/turn the background to <background>}. The new background should be reasonable with objects. Don't include any \" in the instruction and response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, the new background, and output caption for you. Let's get started!"]
    elif instruction_type == 'tone_transfer':
        system_message = "You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is JSON with the following string fields: 'edit', 'new state', 'output'. Here is the latest conversation between Assistant and User."
        intro_message = [
            "Hi, My job to take a given caption ('input') and to output the following: an instruction for changing the overall state about weather/time/season in the image ('edit'), the new state ('new state'), the caption with the new state ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'new state': '<state>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['make', 'change', 'turn']. Use the following format to construct instruction: {change/make/turn the weather/time/season to <state>}. The new state should only be about time change, weather change, or season change. Don't include any \" in the instruction and response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, the new state, and output caption for you. Let's get started!"]
    elif instruction_type == 'textual':
        system_message = "You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is JSON with the following string fields: 'edit', 'input', 'output'. Here is the latest conversation between Assistant and User."
        intro_message = [
            "Hi, my job is to modify a given description and text and output the following: an instruction for changing the text ('edit'), the combined description and text as input ('input'), and the modified version as output ('output'). Please help me do it. When you reply, use the following format: {'edit': '<instruction>', 'input': '<caption>', 'output': '<caption>'}. Construct the instruction using one of the following words: ['alter', 'change', 'replace', 'turn'] . You should ensure the number of words of the text remains the same before and after the change.",
            "Sure, I'd be happy to help! Just provide me with a 'description' and 'text', and I'll generate the instruction, input, and output for you. Let's get started!"
        ]
    elif instruction_type == 'visual_reference':
        # 不是直接产生相关指令，相关指令用remove和replace来转化，输入的input是现在这个的output
        # input: a cow is sitting on a grass ; output: a cow and a rabbit are sitting on a grass
        # visual reference的input和output都是自己产生的，input保留一个object在一个场景，output加上一个新的合理object（比较大的对象）
        return
    else:
        print(f'Error Type {instruction_type}')
        raise NotImplementedError
    # shuffling #
    random.seed(torch.randint(1 << 32, ()).item())
    random.shuffle(few_shot_examples) #  shuffling between examples
    #  few_shot_examples: randomly sampling 60% of the examples
    examples = []
    for e in few_shot_examples[:5]:
        examples += e
    prompt = process_text_multi_turn(history=intro_message+examples, text=new_prompt,
                                     tokenizer=tokenizer, system_prompt=system_message)

    return prompt

def instruction_evaluation(instruction, instruction_type, tokenizer):
    # system message #
    system_message = f"You are an assistant that only speaks Yes or No. Do not write other text. Your job is to help me determine whether an instruction is of a given type. Instruction types are included in ['add', 'remove', 'replace', 'color alter', 'appearance alter', 'action change', 'background change', 'tune transfer', 'implicit']. Here is the latest conversation between Assistant and User."
    
    if instruction_type == 'add':
        intro_message = ["Hi, My job is to determine whether an instruction is the type of 'add' instruction. Please help me do it. The 'add' command must add an object.",
                         "Sure, I'd be happy to help! Just provide me with the instruction."]
        example_message = ["Is 'add a hat to the cat' a 'add' type instruction?", "Yes.",
                           "Is 'replace the shirt to a coat' a 'add' type instruction?","No."]
    elif instruction_type == 'remove':
        intro_message = ["Hi, My job is to determine whether an instruction is the type of 'remove' instruction. Please help me do it. The 'remove' command must remove an object.",
                         "Sure, I'd be happy to help! Just provide me with the instruction."]
        example_message = ["Is 'erase the bird by the yellow flower' a 'remove' type instruction?", "Yes.",
                           "Is 'replace the shirt to a coat' a 'remove' type instruction?", "No."]
    elif instruction_type == 'replace':
        intro_message = ["Hi, My job is to determine whether an instruction is the type of 'replace' instruction. Please help me do it. The 'replace' command must replace an object with a new object. Please replace object not people related",
                         "Sure, I'd be happy to help! Just provide me with the instruction."]
        example_message = ["Is 'alter the cat to dog' a 'replace' type instruction?", "Yes.",
                           "Is 'alter the color of cat to black' a 'replace' type instruction?", "No.",
                           "Is 'remove the cat' a 'replace' type instruction?", "No.",
                           "Is 'replace the hat from the person's head with a  mask' a 'replace' type instruction?", "Yes.",
                           "Is 'replace the driver with a chef' a 'replace' type instruction?", "No.",
                           "Is 'replace the child with a adult' a 'replace' type instruction?", "No."]
    elif instruction_type == 'color_alter':  # input needs color will be checked in pre-fliter
        intro_message = ["Hi, My job is to determine whether an instruction is the type of 'color alter' instruction. Please help me do it. The 'color alter' command must change the color of an object instead of other attributes.",
                         "Sure, I'd be happy to help! Just provide me with the instruction."]
        example_message = ["Is 'alter the color of cat to black' a 'color alter' type instruction?", "Yes.",
                           "Is 'change the color of transportation system to bright' a 'color alter' type instruction?", "No."]
    elif instruction_type == 'appearance_alter':
        intro_message = ["Hi, My job is to determine whether an instruction is the type of 'appearance alter' instruction. Please help me do it. Notice that color and material changing is not belong to 'appearance alter' command and the modifications should be specific, not some abstract changes",
                         "Sure, I'd be happy to help! Just provide me with the instruction."]
        example_message = ["Is 'make the cake decorated with candles' a 'appearance alter' type instruction?", "Yes.",
                           "Is 'change the transportation system to bright' a 'appearance alter' type instruction?", "No.",
                           "Is 'turn the material of chair to wooden' a 'appearance alter' type instruction?", "No.",
                           "Make your dining table more modern' a 'appearance alter' type instruction?", "No.",
                           "Is 'alter the color of cat to black' a 'appearance alter' type instruction?", "No."]
    elif instruction_type == 'material_alter':
        intro_message = [
            "Hi, My job is to determine whether an instruction is the type of 'material alter' instruction. Please help me do it. The 'material alter' command must change the material of an object, including texture, material.",
            "Sure, I'd be happy to help! Just provide me with the instruction."]
        example_message = ["Is 'make the cake decorated with candles' a 'appearance alter' type instruction?", "No.",
                           "Is 'change the transportation system to bright' a 'appearance alter' type instruction?","No.",
                           "Is 'turn the material of chair to wooden' a 'appearance alter' type instruction?", "Yes.",
                           "Is 'alter the color of cat to black' a 'appearance alter' type instruction?", "No."]

    elif instruction_type == 'action_change':
        intro_message = ["Hi, My job is to determine whether an instruction is the type of 'action change' instruction. Please help me do it. The 'action change' command must change the action instead of others, and the action must be reasonable.", "Sure, I'd be happy to help! Just provide me with the instruction."]
        example_message = ["Is 'make the dog lie on the bed' a logical 'action change' type instruction?", "Yes.",
                           "Is 'change the cat to jump' a logical 'action change' type instruction?", "Yes.",
                           "Is 'change cat to black' a logical 'action change' type instruction?", "No.",
                           "Is 'change bowl to run' a logical 'action change' type instruction?", "No."]
    elif instruction_type == 'background_change':
        intro_message = ["Hi, My job is to determine whether an instruction is the type of 'background change' instruction. Please help me do it. The 'background change' instruction must change only the background of the image and should be reasonable with objects.",
                         "Sure, I'd be happy to help! Just provide me with the instruction."]
        example_message = ["Is 'change the background to a beach' a 'background change' type instruction?", "Yes.",
                           "Is 'make the cake decorated with candles' a 'background change' type instruction?", "No."]
    elif instruction_type == 'tone_transfer':
        intro_message = ["Hi, My job is to determine whether an instruction is the type of 'tune transfer' instruction. Please help me do it. The 'tune transfer' command must change the overall state of the image, which can only be time, season, or weather.",
                         "Sure, I'd be happy to help! Just provide me with the instruction."]
        example_message = ["Is 'Change the background to a beach' a 'tune transfer' type instruction?", "No.",
                           "Is 'Change the weather to a rainy day' a 'tune transfer' type instruction?", "Yes.",
                           "Is 'make the day to midnight' a 'tune transfer' type instruction?", "Yes."]
    elif instruction_type == 'textual':
        intro_message = [
            "Hi, My job is to determine whether an instruction is the type of 'textual' instruction. Please help me do it. The 'textual' command must change the text in the image while ensuring the number of words remains the same before and after the change.",
            "Sure, I'd be happy to help! Just provide me with the instruction."] # 修改前后长度一致更容易满足一致性
        example_message = [
            "Is 'Change the text 'oh good' to 'Hello world'' a 'textual' type instruction?", "Yes.",
            # "Is 'Change the text 'Deep Learning' to 'Learning'' a 'textual' type instruction?", "No.",
            "Is 'alter the cat to dog' a 'textual' type instruction?", "No.",
            "Is 'alter the color of cat to black' a 'textual' type instruction?", "No.",
            "Is 'remove the cat' a 'textual' type instruction?", "No."
        ]
    else:
        print(f'Error Type {instruction_type}')
        raise NotImplementedError

    prompt = process_text_multi_turn(history=intro_message+example_message,
                                     text=f"Is '{instruction}' a '{instruction_type}' type instruction?",
                                     tokenizer=tokenizer, system_prompt=system_message)

    return prompt


def generate_tags(raw_text):
    # generate specific categories in the caption by spacy
    nlp = spacy.load("en_core_web_sm")
    tags = {'nouns':[], 'adj':[], 'verb': []}
    words_list = nlp(raw_text)
    for i in range(len(words_list)-1):
        token = words_list[i]
        next_token = words_list[i+1]
        if token.pos_ == 'ADJ' and next_token.pos_ == 'NOUN':
            tags['adj'].append(token.text.lower())
        elif token.pos_ == 'NOUN' and next_token.pos_ != 'ADP':
            tags['nouns'].append(token.text.lower())
        elif token.pos_ == 'VERB':
            tags['verb'].append(token.text.lower())
    if words_list[-1].pos_ == 'NOUN':
        tags['nouns'].append(words_list[-1].text.lower())
    return tags