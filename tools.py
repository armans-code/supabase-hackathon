import base64
import cv2
import dbutils
from PIL import Image
import threading
import anthropic
import time
from cartesia import Cartesia
from elevenlabs import play, VoiceSettings, stream
from elevenlabs.client import ElevenLabs


client = anthropic.Anthropic()
voice = ElevenLabs(api_key="")


def get_tool(user_query: str):
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        system="""You're a tool selector tool that finds the best tool for the user query. You should return the name of the tool that should be used for the user query. The tools available are: use_current_image, recall_previous_image, use_loop.
        
        - The use_current_image tool should be used when the user query is asking about a visual task that can be answered with the current image in the present. For example, "What do you see in front of you?", or "Solve this equation", or "What is the color of the object in front of you?".
        - The recall_previous_image tool should be used when the user query is asking about a visual task that can be answered with the previous image. For example, "I can't find my keys, do you remember where I left them?", or "What color was the triangle from the image I just showed you?".
        - The use_loop tool should be used when the user query is asking about a visual task that can be answered with a sequence of images. For example, "Tell me when you see a red car", or "Tell me when you see an animal that may be living in the water".


        Return the use_loop tool if the user needs to wait for you to find something in the image. For example, return the use_loop tool if the user says "Tell me when you see a blue pen"
        
        Remember to only return the name of the tool that should be used, and nothing else. Return the exact name of the tool. You must return either use_current_image, recall_previous_image, or use_loop.""",
        messages=[
            {
                "role": "user",
                "content": f"""Which tool should be used for the user query? user query: {user_query}
             """,
            }
        ],
    )
    return response.content[0].text


def use_current_image(user_query: str):
    with open("latest.jpeg", "rb") as image_file:
        image_data = image_file.read()

    base64_image = base64.b64encode(image_data).decode("utf-8")

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        system="""You're a helpful assistant that processes the user query using the current image. You should return the answer to the user query based on the current image. Do not mention the image or mention that you are looking at an image. Simply respond as if you were a real person, and the image is your eyesight. Be brief and respond with two sentences at most. Remember, do not mention the image or that you are looking at an image. Instead, say "I see..." or "I notice...".""",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                    {"type": "text", "text": f"User query: {user_query}"},
                ],
            }
        ],
    )

    return response.content[0].text


def use_loop(user_query: str):
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        system="""You analyze user requests to describe what to detect in an image. You must return what the stop condition is for an agentic AI loop.
             Your job is only to determine what the condition is that the AI should look for in an image.
             DO NOT RETURN ANYTHING, EXCEPT FOR THE STOP CONDITION. ONLY RESPOND WITH THE STOP CONDITION AND NOTHING ELSE.
             THE STOP CONDITION MUST BE A STATEMENT THAT CAN BE EITHER TRUE OR FALSE BASED ON AN IMAGE.

             EXAMPLE USER INPUT: "tell me when you see something blue"
             EXAMPLE OUTPUT: "there is a blue object in the image."

             EXAMPLE USER INPUT: "find a cat"
             EXAMPLE OUTPUT: "a cat is in this image."
             
             Remember, you must only return the stop condition based on the user query. Do not return anything else.""",
        messages=[
            {
                "role": "user",
                "content": f"User query: {user_query}",
            }
        ],
    )

    stop_condition = response.content[0].text
    print("INFO: stop condition: " + stop_condition)

    while True:
        print("new")
        with open("latest.jpeg", "rb") as image_file:
            image_data = image_file.read()

        base64_image = base64.b64encode(image_data).decode("utf-8")

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system="""You are analyzing an image. You must answer if a condition has been met or not within the supplied image.
            Based on the objects or characteristics in the image, respond with "Yes" or "No". If you respond with "Yes", you must
            also describe what the condition is that has been met and where it is in the image. Otherwise, only respond with "No" and
            nothing else.

            EXAMPLE USER INPUT: "there is a blue object in the image"
            If you see a blue object in the image, you should respond with something like "Yes, there is a blue object in the image. It appears to be a blue water bottle on a desk and appears next to a notebook."
            If you do not see a blue object in the image, you should respond with "No".

            EXAMPLE USER INPUT: "there is a cat in the image"
            If you see a cat in the image, you should respond with something like "Yes, there is a cat in the image. It is sitting on a chair."
            If you do not see a cat in the image, you should respond with "No".
            """,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"condition: {stop_condition}"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
        )
        print(response.content[0].text)
        if "Yes" in response.content[0].text:
            return response.content[0].text
        time.sleep(1)


def use_recall(query_string):
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        system="you're a keyword extractor tool that finds keywords from user input that will be used in searching a vector database. extract the relevant keywords into one string from the user's request. Only return the string itself, do not return anything else. Do not include any punctuation in the string or any symbols. Only include words that will be useful for searching the database for that image. For example, if the user says 'I haven't seen my keys in a while. They are blue and shiny', you should return 'blue shiny keys'",
        messages=[
            {"role": "user", "content": "User input: " + query_string},
        ],
    )
    keywords = response.content[0].text
    print("INFO: keywords: " + keywords)

    img_path = dbutils.search(keywords)
    print("INFO: image path: " + img_path)

    with open(img_path, "rb") as image_file:
        image_data = image_file.read()

    base64_image = base64.b64encode(image_data).decode("utf-8")

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        system="""You're a tool that processes the user query using a previous image. You should return the answer to the user query based on the previous image. Do not mention the image or mention that you are looking at an image. Simply respond as if you were a real person, and the image is your eyesight. Be brief and respond with two sentences at most. Remember, do not mention the image or that you are looking at an image. Instead, say "I saw..." or "Yes, I remember...". Do not mention the image at all.
        
        EXAMPLE USER INPUT: "I haven't see my glasses recently, do you know where they are?"
        EXAMPLE OUTPUT IF SEEN: "Yes, I remember. They are on the table in the living room, next to a blue mug."
        EXAMPLE OUTPUT IF NOT SEEN: "No, I don't remember seeing them."
        """,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                    {"type": "text", "text": f"User query: {query_string}"},
                ],
            }
        ],
    )

    return response.content[0].text


def get_user_input():
    while True:
        user_input = input("Enter command: ")
        if user_input == "exit":
            break

        tool = get_tool(user_input)
        print("INFO: using tool " + tool)
        if "use_current_image" in tool:
            response = use_current_image(user_input)
            print(response)
            res = voice.text_to_speech.convert_as_stream(
                voice_id="pMsXgVXv3BLzUgSXRplE",
                optimize_streaming_latency="0",
                output_format="mp3_22050_32",
                text=response,
                voice_settings=VoiceSettings(
                    stability=0.1,
                    similarity_boost=0.3,
                    style=0.2,
                ),
            )
            stream(res)
            # play(res)
        elif "use_loop" in tool:
            response = use_loop(user_input)
            print(response)
            res = voice.text_to_speech.convert_as_stream(
                voice_id="pMsXgVXv3BLzUgSXRplE",
                optimize_streaming_latency="0",
                output_format="mp3_22050_32",
                text=response,
                voice_settings=VoiceSettings(
                    stability=0.1,
                    similarity_boost=0.3,
                    style=0.2,
                ),
            )
            stream(res)
        elif "recall_previous_image" in tool:
            response = use_recall(user_input)
            print(response)
            res = voice.text_to_speech.convert_as_stream(
                voice_id="pMsXgVXv3BLzUgSXRplE",
                optimize_streaming_latency="0",
                output_format="mp3_22050_32",
                text=response,
                voice_settings=VoiceSettings(
                    stability=0.1,
                    similarity_boost=0.3,
                    style=0.2,
                ),
            )
            stream(res)


def use_user_input(user_input):
    tool = get_tool(user_input)
    print("INFO: using tool " + tool)
    if "use_current_image" in tool:
        response = use_current_image(user_input)
        print(response)
        res = voice.text_to_speech.convert_as_stream(
            voice_id="pMsXgVXv3BLzUgSXRplE",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=response,
            voice_settings=VoiceSettings(
                stability=0.1,
                similarity_boost=0.3,
                style=0.2,
            ),
        )
        stream(res)
        # play(res)
    elif "use_loop" in tool:
        response = use_loop(user_input)
        print(response)
        res = voice.text_to_speech.convert_as_stream(
            voice_id="pMsXgVXv3BLzUgSXRplE",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=response,
            voice_settings=VoiceSettings(
                stability=0.1,
                similarity_boost=0.3,
                style=0.2,
            ),
        )
        stream(res)
    elif "recall_previous_image" in tool:
        response = use_recall(user_input)
        print(response)
        res = voice.text_to_speech.convert_as_stream(
            voice_id="pMsXgVXv3BLzUgSXRplE",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=response,
            voice_settings=VoiceSettings(
                stability=0.1,
                similarity_boost=0.3,
                style=0.2,
            ),
        )
        stream(res)


if __name__ == "__main__":
    # input_thread = threading.Thread(target=get_user_input)
    # input_thread.daemon = True
    # input_thread.start()

    cam = cv2.VideoCapture(0)
    frame_counter = 0
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        # turn frame into base64
        cv2.resize(frame, (480, 270))
        cv2.imshow("frame", frame)

        if frame_counter % 30 == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            threading.Thread(
                target=dbutils.store_frame, args=(img, frame_counter)
            ).start()

        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
