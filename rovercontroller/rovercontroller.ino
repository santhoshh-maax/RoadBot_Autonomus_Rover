// =====================================================
// ESP32 ROVER MOTOR CONTROLLER (JETSON CONTROLLED)
// =====================================================

// ==============================
// MOTOR PIN DEFINITIONS
// ==============================
#define IN1 26
#define IN2 27
#define IN3 14
#define IN4 12

// ==============================
char currentCommand = 'S';
unsigned long lastCommandTime = 0;
const unsigned long safetyTimeout = 3000; // 3 seconds

bool isMoving = false;

// =====================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("=================================");
  Serial.println("ESP32 Rover Ready");
  Serial.println("Waiting for Jetson...");
  Serial.println("Commands: F B L R S");
  Serial.println("=================================");

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  stopCar();
}

// =====================================================
void loop() {

  // ==============================
  // READ COMMAND FROM JETSON
  // ==============================
  if (Serial.available()) {

    char command = Serial.read();

    if (command == '\n' || command == '\r') return;

    currentCommand = command;
    lastCommandTime = millis();

    Serial.print("Command Received: ");
    Serial.println(command);

    switch (command) {

      case 'R':
        forward();
        isMoving = true;
        Serial.println("➡ Moving FORWARD");
        break;

      case 'L':
        backward();
        isMoving = true;
        Serial.println("⬅ Moving BACKWARD");
        break;

      case 'B':
        left();
        isMoving = true;
        Serial.println("↩ Turning LEFT");
        break;

      case 'F':
        right();
        isMoving = true;
        Serial.println("↪ Turning RIGHT");
        break;

      case 'S':
        stopCar();
        isMoving = false;
        Serial.println("STOPPED");
        break;

      default:
        Serial.println("Unknown Command - STOP");
        stopCar();
        isMoving = false;
        break;
    }

    // Send acknowledgement
    Serial.println("ACK");
  }

  // ==============================
  // SAFETY TIMEOUT
  // ==============================
  if (millis() - lastCommandTime > safetyTimeout) {
    if (isMoving) {
      stopCar();
      isMoving = false;
      Serial.println("SAFETY TIMEOUT - AUTO STOP");
    }
  }
}

// =====================================================
void forward() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void backward() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void left() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void right() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void stopCar() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}
