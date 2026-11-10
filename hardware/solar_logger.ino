/*
  Solar PV logger - Arduino Uno / Nano / ESP32

  Wiring (INA219 current/voltage sensor, I2C):
    PV panel (+) -> INA219 VIN+
    INA219 VIN-  -> load resistor (~7-10 ohm, 5 W) -> PV panel (-)
    INA219 GND   -> Arduino GND and PV panel (-)
    INA219 VCC   -> 5V (Uno) / 3.3V (ESP32)
    INA219 SDA   -> A4 (Uno) / GPIO21 (ESP32)
    INA219 SCL   -> A5 (Uno) / GPIO22 (ESP32)

  Optional sensors (set the #define to 1 to enable):
    BH1750 light sensor on the same I2C bus -> light_lux
    DHT11/DHT22 on pin 2                    -> temperature_C

  Output: one CSV line per minute (average of 60 one-second samples)
    voltage_V,current_mA,power_mW,temperature_C,light_lux
  Read it on the PC with:  python serial_logger.py --port COMx

  Libraries (Arduino Library Manager): "Adafruit INA219",
  optionally "BH1750" (Christopher Laws) and "DHT sensor library" (Adafruit).
*/
#include <Wire.h>
#include <Adafruit_INA219.h>

#define USE_BH1750 0
#define USE_DHT    0
#define DHT_PIN    2
#define DHT_TYPE   DHT11

#if USE_BH1750
#include <BH1750.h>
BH1750 lightMeter;
#endif
#if USE_DHT
#include <DHT.h>
DHT dht(DHT_PIN, DHT_TYPE);
#endif

Adafruit_INA219 ina219;

const unsigned long SAMPLE_MS = 1000;   // read sensors every second
const int SAMPLES_PER_LINE = 60;        // print the average every minute

float sumV = 0, sumI = 0, sumP = 0, sumLux = 0;
int n = 0;
unsigned long lastSample = 0;

void setup() {
  Serial.begin(9600);
  Wire.begin();
  if (!ina219.begin()) {
    Serial.println("# ERROR: INA219 not found - check wiring");
    while (1) delay(1000);
  }
  // 32V / 1A range gives better resolution for a small panel
  ina219.setCalibration_32V_1A();
#if USE_BH1750
  lightMeter.begin();
#endif
#if USE_DHT
  dht.begin();
#endif
  Serial.println("# solar logger ready: voltage_V,current_mA,power_mW,temperature_C,light_lux");
}

void loop() {
  if (millis() - lastSample < SAMPLE_MS) return;
  lastSample = millis();

  float busV = ina219.getBusVoltage_V();
  float shuntmV = ina219.getShuntVoltage_mV();
  float current = ina219.getCurrent_mA();
  if (current < 0) current = 0;                 // ignore reverse noise at night
  float loadV = busV + shuntmV / 1000.0;

  sumV += loadV;
  sumI += current;
  sumP += loadV * current;                       // mW
#if USE_BH1750
  sumLux += lightMeter.readLightLevel();
#endif
  n++;

  if (n >= SAMPLES_PER_LINE) {
    float temp = NAN;
#if USE_DHT
    temp = dht.readTemperature();
#endif
    Serial.print(sumV / n, 3);  Serial.print(',');
    Serial.print(sumI / n, 2);  Serial.print(',');
    Serial.print(sumP / n, 1);  Serial.print(',');
    if (!isnan(temp)) Serial.print(temp, 1);
    Serial.print(',');
#if USE_BH1750
    Serial.print(sumLux / n, 0);
#endif
    Serial.println();
    sumV = sumI = sumP = sumLux = 0;
    n = 0;
  }
}
