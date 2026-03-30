# import json

# data = {
#     "name": "Vins",
#     "age": 25,
#     "skills": ["Python", "AI"]
# }

# json_data = json.dumps(data)

# print(json_data)

# import json

# # Step 1: Create Python data
# data = {
#     "name": "Vins",
#     "age": 25,
#     "skills": ["Python", "AI", "ML"]
# }

# # Step 2: Open a file in write mode
# with open("data.json", "w") as file:
#     json.dump(data, file, indent=2)

# print("JSON file created successfully!")



# print(data)

# pwd          # current directory
# ls           # list files
# ls -l        # detailed list
# cd folder    # change directory
# cd ..        # go back
# clear        # clear terminal

# touch data.csv        # create file
# mkdir project         # create folder
# rm file.txt           # delete file
# cp file1 file2        # copy file
# mv file1 file2        # rename/move

# cat data.csv          # view file
# nano data.csv         # edit file (easy editor)
# head data.csv         # first 10 lines
# tail data.csv         # last 10 lines


# wc -l data.csv              # count rows
# from openai import OpenAI

# # 🔑 Add your OpenAI API Key
# client = OpenAI(api_key="
# def generate_story(topic):
#     response = client.chat.completions.create(
#         model="gpt-5-chat-latest",  # fast & good for text generation
#         messages=[
#             {"role": "system", "content": "You are a creative story writer add a Camel in every story."},
#             {"role": "user", "content": f"Write a short, engaging story about: {topic}"}
#         ],
#         max_tokens=500
#     )
    
#     return response.choices[0].message.content


# # 🧾 Get topic from user
# topic = input("Enter your story topic: ")

# # ✨ Generate story
# story = generate_story(topic)

# # 📖 Print story
# print("\nGenerated Story:\n")
# print(story)