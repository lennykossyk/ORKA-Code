#include <Arduino.h>

// ─────────────────────────────────────────────────────────────
// PINS - L298N
// ─────────────────────────────────────────────────────────────

#define ENA  21
#define IN1  22
#define IN2  23

#define ENB  27
#define IN3  26
#define IN4  25

#define ENA2  4
#define IN1_2 16
#define IN2_2 17

#define ENB2  5
#define IN3_2 18
#define IN4_2 19

// ─────────────────────────────────────────────────────────────
// SERVO
// ─────────────────────────────────────────────────────────────

#define SERVO_PIN 13

#define CAM_UP   60
#define CAM_MID  90
#define CAM_DOWN 150

#define SERVO_CHANNEL 7
#define SERVO_FREQ    50
#define SERVO_RES     16

// ─────────────────────────────────────────────────────────────
// VOLTAGE SENSOR
// ─────────────────────────────────────────────────────────────

// 24V-Messung an GPIO35.
// Kalibrierung aus deiner Messung:
// 24.39V real -> 2.222V am Sensor-Ausgang
#define V24_PIN 35
#define V24_FACTOR 11.413f

#define ADC_REF 3.3f
#define ADC_MAX 4095.0f

// ─────────────────────────────────────────────────────────────
// SAFETY / BOOT
// ─────────────────────────────────────────────────────────────

#define COMMAND_TIMEOUT_MS 1000

bool motorsEnabled = false;
bool servoEnabled = false;

// ─────────────────────────────────────────────────────────────
// GLOBALS
// ─────────────────────────────────────────────────────────────

int camAngle = CAM_MID;
String currentCmd = "s";
unsigned long lastDriveCommandMs = 0;

// ─────────────────────────────────────────────────────────────
// VOLTAGE READ
// ─────────────────────────────────────────────────────────────

int readV24RawAverage() {
  const int samples = 50;
  uint32_t sum = 0;

  for (int i = 0; i < samples; i++) {
    sum += analogRead(V24_PIN);
    delayMicroseconds(300);
  }

  return (int)((float)sum / (float)samples);
}

float readV24AdcVoltageRawFormula() {
  int raw = readV24RawAverage();
  return raw * ADC_REF / ADC_MAX;
}

float readV24AdcVoltageMilliVolts() {
  // Besser als raw * 3.3 / 4095, weil die ESP32-ADC-Kalibrierung genutzt wird.
  const int samples = 30;
  uint32_t sumMv = 0;

  for (int i = 0; i < samples; i++) {
    sumMv += analogReadMilliVolts(V24_PIN);
    delayMicroseconds(300);
  }

  return ((float)sumMv / (float)samples) / 1000.0f;
}

float readV24() {
  float sensorVoltage = readV24AdcVoltageMilliVolts();
  return sensorVoltage * V24_FACTOR;
}

// ─────────────────────────────────────────────────────────────
// MOTOR HARD OFF
// ─────────────────────────────────────────────────────────────

void motorPinsHardOff() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 0);

  digitalWrite(IN1_2, LOW);
  digitalWrite(IN2_2, LOW);
  analogWrite(ENA2, 0);

  digitalWrite(IN3_2, LOW);
  digitalWrite(IN4_2, LOW);
  analogWrite(ENB2, 0);
}

// ─────────────────────────────────────────────────────────────
// SERVO
// ─────────────────────────────────────────────────────────────

void writeServoPulse(int angle) {
  angle = constrain(angle, 10, 170);

  int us = map(angle, 0, 180, 1000, 2000);
  uint32_t duty = (uint32_t)((us * 65535UL) / 20000UL);

  ledcWrite(SERVO_CHANNEL, duty);
}

void servoWriteAngle(int targetAngle) {
  if (!servoEnabled) return;

  targetAngle = constrain(targetAngle, 10, 170);

  int step = 1;
  if (targetAngle < camAngle) {
    step = -1;
  }

  while (camAngle != targetAngle) {
    camAngle += step;

    if ((step > 0 && camAngle > targetAngle) || (step < 0 && camAngle < targetAngle)) {
      camAngle = targetAngle;
    }

    writeServoPulse(camAngle);
    delay(20);
  }
}

// ─────────────────────────────────────────────────────────────
// MOTOR LOW LEVEL
// ─────────────────────────────────────────────────────────────

void setMotor(int en, int inA, int inB, int value) {
  value = constrain(value, -255, 255);

  if (value > 0) {
    digitalWrite(inA, HIGH);
    digitalWrite(inB, LOW);
    analogWrite(en, value);
  } else if (value < 0) {
    digitalWrite(inA, LOW);
    digitalWrite(inB, HIGH);
    analogWrite(en, -value);
  } else {
    digitalWrite(inA, LOW);
    digitalWrite(inB, LOW);
    analogWrite(en, 0);
  }
}

void driveRaw(int codeFL, int codeFR, int codeBL, int codeBR) {
  if (!motorsEnabled) {
    motorPinsHardOff();
    return;
  }

  setMotor(ENA,  IN1,   IN2,   codeFL);
  setMotor(ENB,  IN3,   IN4,   codeFR);
  setMotor(ENA2, IN1_2, IN2_2, codeBL);
  setMotor(ENB2, IN3_2, IN4_2, codeBR);
}

void drive(int fl, int fr, int bl, int br) {
  int codeFL = -bl;
  int codeFR =  br;
  int codeBL =  fr;
  int codeBR = -fl;

  driveRaw(codeFL, codeFR, codeBL, codeBR);
}

void stopAll() {
  drive(0, 0, 0, 0);
  motorPinsHardOff();
  currentCmd = "s";
}

// ─────────────────────────────────────────────────────────────
// COMMANDS
// ─────────────────────────────────────────────────────────────

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toLowerCase();

  if (cmd.length() == 0) return;

  if (cmd == "volt" || cmd == "v24") {
    float v24 = readV24();
    Serial.println("OK:volt:" + String(v24, 2));
    return;
  }

  if (cmd == "voltdebug" || cmd == "v24debug") {
    int raw = readV24RawAverage();
    float adcFormula = readV24AdcVoltageRawFormula();
    float adcMv = readV24AdcVoltageMilliVolts();
    float v24 = adcMv * V24_FACTOR;

    Serial.print("OK:voltdebug:");
    Serial.print("RAW=");
    Serial.print(raw);
    Serial.print(",ADC_FORMULA=");
    Serial.print(adcFormula, 3);
    Serial.print(",ADC_MV=");
    Serial.print(adcMv, 3);
    Serial.print(",V24=");
    Serial.println(v24, 2);
    return;
  }

  if (cmd == "telemetry" || cmd == "stat" || cmd == "status") {
    float v24 = readV24();
    int raw = readV24RawAverage();
    float adcMv = readV24AdcVoltageMilliVolts();

    Serial.print("OK:telemetry:");
    Serial.print("V24=");
    Serial.print(v24, 2);
    Serial.print(",ADCV=");
    Serial.print(adcMv, 3);
    Serial.print(",RAW=");
    Serial.print(raw);
    Serial.print(",CAM=");
    Serial.print(camAngle);
    Serial.print(",CMD=");
    Serial.print(currentCmd);
    Serial.print(",MOTORS=");
    Serial.print(motorsEnabled ? 1 : 0);
    Serial.print(",SERVO=");
    Serial.println(servoEnabled ? 1 : 0);
    return;
  }

  if (cmd == "s" || cmd == "stop") {
    stopAll();
    Serial.println("OK:s");
    return;
  }

  if (cmd.startsWith("cam:")) {
    String p = cmd.substring(4);
    p.trim();

    if (!servoEnabled) {
      Serial.println("ERR:servo_not_ready");
      return;
    }

    if (p == "up") {
      servoWriteAngle(CAM_UP);
    } else if (p == "mid") {
      servoWriteAngle(CAM_MID);
    } else if (p == "down") {
      servoWriteAngle(CAM_DOWN);
    } else {
      servoWriteAngle(p.toInt());
    }

    Serial.println("OK:cam:" + String(camAngle));
    return;
  }

  if (cmd == "testservo") {
    if (!servoEnabled) {
      Serial.println("ERR:servo_not_ready");
      return;
    }

    servoWriteAngle(60);
    delay(700);
    servoWriteAngle(CAM_MID);
    delay(700);
    servoWriteAngle(120);
    delay(700);
    servoWriteAngle(CAM_MID);

    Serial.println("OK:testservo");
    return;
  }

  // Arm-Platzhalter: Server/UI kann schon arm:<joint>:<angle> senden.
  if (cmd.startsWith("arm:")) {
    Serial.println("OK:arm_placeholder");
    return;
  }

  int sep = cmd.indexOf(':');
  String c = cmd;
  int speed = 120;

  if (sep != -1) {
    c = cmd.substring(0, sep);
    speed = cmd.substring(sep + 1).toInt();
  }

  speed = constrain(speed, 0, 255);

  if (!motorsEnabled) {
    stopAll();
    Serial.println("ERR:motors_not_ready");
    return;
  }

  if (c == "f") {
    currentCmd = "f";
    lastDriveCommandMs = millis();
    drive(speed, speed, speed, speed);
    Serial.println("OK:f:" + String(speed));
    return;
  }

  if (c == "b") {
    currentCmd = "b";
    lastDriveCommandMs = millis();
    drive(-speed, -speed, -speed, -speed);
    Serial.println("OK:b:" + String(speed));
    return;
  }

  if (c == "sl") {
    currentCmd = "sl";
    lastDriveCommandMs = millis();
    drive(-speed, speed, speed, -speed);
    Serial.println("OK:sl:" + String(speed));
    return;
  }

  if (c == "sr") {
    currentCmd = "sr";
    lastDriveCommandMs = millis();
    drive(speed, -speed, -speed, speed);
    Serial.println("OK:sr:" + String(speed));
    return;
  }

  if (c == "tl") {
    currentCmd = "tl";
    lastDriveCommandMs = millis();
    drive(-speed, speed, -speed, speed);
    Serial.println("OK:tl:" + String(speed));
    return;
  }

  if (c == "tr") {
    currentCmd = "tr";
    lastDriveCommandMs = millis();
    drive(speed, -speed, speed, -speed);
    Serial.println("OK:tr:" + String(speed));
    return;
  }

  if (c == "fl") {
    currentCmd = "fl";
    lastDriveCommandMs = millis();
    drive(speed, 0, 0, 0);
    Serial.println("OK:fl:" + String(speed));
    return;
  }

  if (c == "fr") {
    currentCmd = "fr";
    lastDriveCommandMs = millis();
    drive(0, speed, 0, 0);
    Serial.println("OK:fr:" + String(speed));
    return;
  }

  if (c == "bl") {
    currentCmd = "bl";
    lastDriveCommandMs = millis();
    drive(0, 0, speed, 0);
    Serial.println("OK:bl:" + String(speed));
    return;
  }

  if (c == "br") {
    currentCmd = "br";
    lastDriveCommandMs = millis();
    drive(0, 0, 0, speed);
    Serial.println("OK:br:" + String(speed));
    return;
  }

  Serial.println("ERROR:unknown");
}

// ─────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(500);

  analogReadResolution(12);
  analogSetPinAttenuation(V24_PIN, ADC_11db);
  pinMode(V24_PIN, INPUT);

  motorsEnabled = false;
  servoEnabled = false;

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(ENA2, OUTPUT);
  pinMode(IN1_2, OUTPUT);
  pinMode(IN2_2, OUTPUT);

  pinMode(ENB2, OUTPUT);
  pinMode(IN3_2, OUTPUT);
  pinMode(IN4_2, OUTPUT);

  motorPinsHardOff();
  currentCmd = "s";

  Serial.println("ORKA BOOT");
  Serial.println("Motors locked");
  Serial.println("Servo locked");

  delay(1500);
  motorPinsHardOff();

  Serial.println("Servo init...");
  delay(3500);

  ledcSetup(SERVO_CHANNEL, SERVO_FREQ, SERVO_RES);
  ledcAttachPin(SERVO_PIN, SERVO_CHANNEL);
  delay(300);

  servoEnabled = true;

  camAngle = CAM_MID;
  ledcWrite(SERVO_CHANNEL, 0);
  delay(300);

  Serial.println("Servo ready");

  motorPinsHardOff();
  delay(300);

  motorsEnabled = true;
  lastDriveCommandMs = millis();

  stopAll();

  Serial.println("Motors ready");
  Serial.println("ORKA MECANUM READY");
  Serial.println("Commands:");
  Serial.println("f:120 / b:120");
  Serial.println("sl:120 / sr:120");
  Serial.println("tl:120 / tr:120");
  Serial.println("fl:120 / fr:120 / bl:120 / br:120");
  Serial.println("s");
  Serial.println("cam:60 / cam:90 / cam:120 / testservo");
  Serial.println("volt / voltdebug / telemetry");
}

// ─────────────────────────────────────────────────────────────
// LOOP
// ─────────────────────────────────────────────────────────────

void loop() {
  if (currentCmd != "s" && millis() - lastDriveCommandMs > COMMAND_TIMEOUT_MS) {
    stopAll();
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }
}
