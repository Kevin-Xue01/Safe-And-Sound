//BME 261 Component 2
// LCD Monitor, Speaker Alarm, LED Alarm, and Bluetooth(?)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//LCD library
#include <LiquidCrystal.h>
#include"pitches.h"

int Contrast=50;
int led=10;
const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

//Design 1
int melody1[] = {
NOTE_D4, NOTE_G4, NOTE_D4, NOTE_G4,
NOTE_D4, NOTE_G4, NOTE_D4, NOTE_G4,
NOTE_D4, NOTE_G4, NOTE_D4, NOTE_G4,
NOTE_D4, NOTE_G4, NOTE_D4, NOTE_G4,
0,END                                 
};

//Design 2
int melody2[] = {
NOTE_C7, NOTE_D8, NOTE_C7, NOTE_D8,
NOTE_C7, NOTE_D8, NOTE_C7, NOTE_D8,
NOTE_C7, NOTE_D8, NOTE_C7, NOTE_D8,
NOTE_C7, NOTE_D8, NOTE_C7, NOTE_D8,
0,END                                 
};

//Design 3
int melody3[] = {
NOTE_A2, NOTE_A2, NOTE_A2,NOTE_A2,
NOTE_A2, NOTE_A2, NOTE_A2,NOTE_A2,
NOTE_A2, NOTE_A2, NOTE_A2,NOTE_A2,
NOTE_A2, NOTE_A2, NOTE_A2,NOTE_A2,
0                                 
};

//Design 4
int melody4[] = { 
NOTE_G5, NOTE_C4, NOTE_G5, NOTE_C4,
NOTE_G5, NOTE_C4, NOTE_G5, NOTE_C4,
NOTE_G5, NOTE_C4, NOTE_G5, NOTE_C4,
NOTE_G5, NOTE_C4, NOTE_G5, NOTE_C4,
0                          
};

// note durations: 8 = quarter note, 4 = 8th note, etc.
int noteDurations1[] = {       //duration of the notes
8,4,8,4,
4,4,4,12,
8,4,8,4,
4,4,4,12
};

int noteDurations2[] = {       //duration of the notes
8,4,8,4,
8,4,8,4,
8,4,8,4,
8,4,8,4,
};

int noteDurations3[] = {       //duration of the notes
8,8,8,8,
8,8,8,8,
8,8,8,8,
8,8,8,8,
8
};

int noteDurations4[] = {       //duration of the notes
8,8,8,8,
8,8,8,8,
8,8,8,8,
8,8,8,8,
8
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int melodyState = 0; 
int prevState = -2; 
int melodySize = 17;

int speed=90;  //higher value, slower notes

void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN,OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  //initialize led pin as an output and its dimensions (width and height)
  pinMode(led, OUTPUT);
  analogWrite(6,Contrast);
  lcd.begin(16, 2);

  lcd.setCursor(0, 0);

}

void loop(){
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    Serial.println(data.toInt());
    switch(data.toInt()) {
      case -2: //Ball not detected
        if(prevState != -2){
          prevState = -2;
          melodyState = 0;
          digitalWrite(led,LOW);
          digitalWrite(LED_BUILTIN, HIGH);
          lcd.setCursor(0,1);
          lcd.print("NOT FOUND ");
          noTone(9);
        }
        
        break;
        
      case 0: //Ball in green zone
        if(prevState != 0){
          prevState = 0;
          melodyState = 0;
          digitalWrite(led,LOW);
          digitalWrite(LED_BUILTIN, LOW);
          lcd.setCursor(0,1);
          lcd.print("NORMAL   ");
          noTone(9);
        }

        
        break;
        
      case -1: //Ball in red zone; too low
        sound();
        if(prevState == -1){
          melodyState = melodyState + 1;
        } else {
          prevState = -1;
          digitalWrite(led,HIGH);
          lcd.setCursor(0,1);
          lcd.print("TOO LOW ");
        }
        
        break;

      case 1: //Ball in red zone; too high
        
        sound();
        if(prevState == 1){
          melodyState = melodyState + 1;
        } else {
          prevState = 1;
          digitalWrite(led,HIGH);
          lcd.setCursor(0,1);
          lcd.print("TOO HIGH ");
        }
        
        break;
    }
  } 

  
}
  

//sound function to play the alarm; change the number to change the melody
 void sound(){
  int noteDuration = speed*noteDurations4[melodyState];
  tone(9, melody4[melodyState],noteDuration*.95);   
  if(melodyState + 1 >= melodySize){
    melodyState = 0;
  }
 }
