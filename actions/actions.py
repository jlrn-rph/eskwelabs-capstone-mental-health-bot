# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

from typing import Any, Dict, List, Text, Union
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
import numpy as np
import pickle
import requests

class PredictMentalHealthRiskAction(Action):
    def name(self) -> Text:
        return "action_predict_mental_health_risk"

    def __init__(self):
        self.depression_vectorizer = None
        self.anxiety_vectorizer = None

    def preprocess_text(self, text, vectorizer):
        # Transform the text data using the vectorizer
        text_tfidf = vectorizer.transform([text])
        return text_tfidf

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Load the depression model and vectorizer
        depression_path = './models-binary/xgboost_for_depression_with_vectorizer.pkl'
        with open(depression_path, 'rb') as f:
            depression_objects = pickle.load(f)

        self.depression_vectorizer = depression_objects[0]
        depression_model = depression_objects[1]

        # Load the anxiety model and vectorizer
        anxiety_path = './models-binary/xgboost_for_anxiety_with_vectorizer.pkl'
        with open(anxiety_path, 'rb') as f:
            anxiety_objects = pickle.load(f)

        self.anxiety_vectorizer = anxiety_objects[0]
        anxiety_model = anxiety_objects[1]

        # Retrieve user input from tracker
        user_input = tracker.latest_message.get("text")

        # Preprocess the user input using the appropriate vectorizers
        depression_text_tfidf = self.preprocess_text(user_input, self.depression_vectorizer)
        anxiety_text_tfidf = self.preprocess_text(user_input, self.anxiety_vectorizer)
        anxiety_labels = ["not likely to be anxious", "anxious"]
        depression_labels = ["not likely to be depressed", "depressed"]

        # Use the models to predict the class probabilities
        anxiety_probs = anxiety_model.predict_proba(anxiety_text_tfidf)
        depression_probs = depression_model.predict_proba(depression_text_tfidf)

        # Get the highest probability index for anxiety
        anxiety_highest_prob_index = np.argmax(anxiety_probs)
        anxiety_highest_prob = anxiety_probs[0, anxiety_highest_prob_index]
        anxiety_label = anxiety_labels[anxiety_highest_prob_index]
        # print(f"Label: {anxiety_label}\nProbability: {anxiety_highest_prob}")

        # Get the highest probability index for depression
        depression_highest_prob_index = np.argmax(depression_probs)
        depression_highest_prob = depression_probs[0, depression_highest_prob_index]
        depression_label = depression_labels[depression_highest_prob_index]
        # print(f"Label: {depression_label}\nProbability: {depression_highest_prob}")

        # Determine the emotional state label based on probabilities
        if anxiety_highest_prob > depression_highest_prob:
            emotional_state_label = anxiety_label
            emotional_state_prob = anxiety_highest_prob
        elif depression_highest_prob > anxiety_highest_prob:
            emotional_state_label = depression_label
            emotional_state_prob = depression_highest_prob

        print(f"Predicted Class: {emotional_state_label}")
        print(f"Probability: {round(emotional_state_prob * 100, 2)}")

        # Set the predicted class and probability in the slots
        predicted_class_slot = SlotSet("predicted_class", emotional_state_label)
        probability_slot = SlotSet("probability", round(emotional_state_prob * 100, 2))

        print(f"{predicted_class_slot}, {probability_slot}")

        return [predicted_class_slot, probability_slot]

class UtterSimilarThoughtsProbability(Action):
    def name(self) -> Text:
        return "action_utter_similar_thoughts_probability"
    
    def format_response(self, dispatcher: CollectingDispatcher, predicted_class: str, probability: float) -> None:
        response = f"It seems your thoughts are {probability}% similar to those of those who are {predicted_class}.\n\n"

        if predicted_class == "depressed":
            resources = """I'm sorry to hear that you're feeling depressed. Remember to take care of yourself and reach out to your support system. If you need professional help, don't hesitate to seek it. You're not alone in this journey. Here are some resources that could help ease your feeling now:\n\n"""
            resources += """📖 [How to cope with depression](https://www.nhs.uk/mental-health/self-help/tips-and-support/cope-with-depression/): Try these coping strategies if you are feeling depressed or have a low mood.\n\n"""
            resources += """📖 [Tips to Manage Depression](https://adaa.org/understanding-anxiety/depression/tips): Get 12 tips from experts you can take to cope with depression.\n\n"""
            resources += """📖 [Depression: What you should know](https://cdn.who.int/media/docs/default-source/campaigns-and-initiatives/world-mental-health-day/2021/1_depression_2021.pdf?sfvrsn=158200d3_17): This is an information sheet briefly explaining depression and providing you with tips to prevent and address depression.\n\n"""
            resources += """📖 [Life with Depression](https://mhanational.org/infographic-life-depression): Read about living with depression and ways to tackle it in everyday situations."""
            response += resources

        elif predicted_class == "anxious":
            resources = """I'm sorry to hear that you're feeling anxious. Remember to practice relaxation techniques such as deep breathing and mindfulness. Engaging in activities you enjoy can also help reduce anxiety. If you need professional help, consider reaching out to a therapist or counselor. You're not alone in this journey. Here are some resources that could help ease your feeling now:\n\n"""
            resources += """📖 [Overcoming Anxiety](https://nicabm-stealthseminar.s3.amazonaws.com/Infographics/Overcoming+Anxiety/NICABM-InfoG-Overcoming-Anxiety.jpg): Learn how to build a values-based mindset to stop anxiety from keeping you from meaningful moments in life.\n\n"""
            resources += """📖 [4 Core Strategies for Managing Anxiety and Stress](https://nicabm-stealthseminar.s3.amazonaws.com/Infographics/Anxiety/NICABM-InfoG-4strateies-for-managing-anxiety.jpg): Learn about Attention-Centering Techniques, Expressive and Creative Strategies, Reflection Exploration Strategies, and Healthy Lifestyle Values.\n\n"""
            resources += """📖 [Stress and Anxiety Tips](https://adaa.org/sites/default/files/downloads/Stress%20and%20Anxiety%20Tips.pdf): Here are 14 ways you can deal with stress and anxiety, ranging from changes in your physical behaviors to your mindsets."""
            response += resources

        else:
            resources = """I'm here to listen and support you. Remember, it's okay to not be okay sometimes. Take care of yourself and reach out to loved ones if you need someone to talk to. Here are some resources that could explore to improve your mental wellbeing:\n\n"""
            resources += """📖 [A Seed Called Courage](https://bit.ly/dohmhphselfcarekit-1): We hope that through this workbook, ay maramdaman niyo na you are not alone. Andito kami para damayan ka as you walk through life. This is a self-care workbook by MentalHealthPH. You may visit their [website](https://mentalhealthph.org/) for more mental health resources. You can use this any time, in any way you want.\n\n"""
            resources += """📖 [Caring for Your Mental Health](https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health): Even small acts of self-care in your daily life can have a big impact on maintaining your mental health and help support your treatment if you have a mental illness. \n\n"""
            resources += """📖 [How to be happier](https://www.nhs.uk/mental-health/self-help/tips-and-support/how-to-be-happier/): Try our 6 tips to help you be happier, more in control, and able to cope better with life's ups and downs.\n\n"""
            resources += """📖 [Mindfulness](https://www.nhs.uk/mental-health/self-help/tips-and-support/mindfulness/): Mindfulness can help mental well-being. Learn how it does so and how you can get started to be more mindful."""
            response += resources

        dispatcher.utter_message(response)


    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        predicted_class = tracker.get_slot("predicted_class")
        probability = tracker.get_slot("probability")

        self.format_response(dispatcher, predicted_class, probability)
        return []

def geocode_address(address):
    geolocator = Nominatim(user_agent="mental_health_facility_geocoder")
    location = geolocator.geocode(address, exactly_one=True, timeout=10)
    if location:
        return location.latitude, location.longitude
    return None, None
    
class NearestMentalHealthFacilityAction(Action):
    def name(self) -> Text:
        return "action_nearest_mental_health_facility"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the user's location from the slot
        location = next(tracker.get_latest_entity_values("location"), None)
        print(location)
        query = f"{location}, Philippines"

        if location:
            # Geocode the user's location
            latitude, longitude = geocode_address(location)

            if latitude is None or longitude is None:
                dispatcher.utter_message("Unable to geocode the location.")
                return []

            # Make a request to the Nominatim API
            response = requests.get(f"https://nominatim.openstreetmap.org/search?q={query}&format=json")

            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()

                if data:
                    # Get the latitude and longitude of the user's location
                    user_latitude = float(data[0]['lat'])
                    user_longitude = float(data[0]['lon'])

                    # Read the mental health facility details from the Excel file
                    facility_data = pd.read_excel('./dataset/mh-ph-facility.xlsx')

                    # Calculate the distance to each mental health facility
                    facility_data['DIST'] = facility_data.apply(lambda row: geodesic((user_latitude, user_longitude), (row['LAT'], row['LONG'])).kilometers, axis=1)

                    if facility_data.empty:
                        dispatcher.utter_message("No mental health facility found for the specified location.")
                    else:
                        # Find the nearest mental health facility
                        nearest_facility = facility_data.loc[facility_data['DIST'].idxmin()]

                        # Get the details of the nearest facility
                        name = nearest_facility['NAME OF HOSPITAL']
                        address = nearest_facility['ADDRESS']
                        contact = nearest_facility['CONTACT']

                        # Create the response message
                        message = f"The nearest mental health facility to your location is {name}. It is located at {address}. You can contact them at {contact}."

                        dispatcher.utter_message(message)
                else:
                    dispatcher.utter_message("No mental health facility found for the specified location.")
            else:
                dispatcher.utter_message("Unable to geocode the location.")
        else:
            dispatcher.utter_message("No location provided.")
            return []