## happy path
* greet
    - utter_greet
* request_restaurant
    - activate_restaurant_form
* thankyou
    - utter_noworries

## unhappy path
* greet
    - utter_greet
* request_restaurant
    - activate_restaurant_form
* chitchat
    - utter_chitchat
    - activate_restaurant_form
* thankyou
    - utter_noworries

## unhappy path
* greet
    - utter_greet
* request_restaurant
    - activate_restaurant_form
* chitchat
    - utter_chitchat
    - activate_restaurant_form
* chitchat
    - utter_chitchat
    - activate_restaurant_form
* chitchat
    - utter_chitchat
    - activate_restaurant_form
* thankyou
    - utter_noworries

## bot challenge
* bot_challenge
  - utter_iamabot

## stop but continue path
* greet
    - utter_greet
* request_restaurant
    - activate_restaurant_form
* stop
    - utter_ask_continue
* affirm
    - activate_restaurant_form
* thankyou
    - utter_noworries
