#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "fb_gfx.h"

#include "face_detector.h"
#include "secrets.h"
#include "web_page.h"

namespace camera_pins {
constexpr int pwdn = -1;
constexpr int reset = -1;
constexpr int xclk = 10;
constexpr int sda = 40;
constexpr int scl = 39;
constexpr int d7 = 48;
constexpr int d6 = 11;
constexpr int d5 = 12;
constexpr int d4 = 14;
constexpr int d3 = 16;
constexpr int d2 = 18;
constexpr int d1 = 17;
constexpr int d0 = 15;
constexpr int vsync = 38;
constexpr int href = 47;
constexpr int pclk = 13;
}

struct RuntimeStats {
  volatile uint32_t faces = 0;
  volatile uint32_t candidates = 0;
  volatile uint32_t inference_ms = 0;
  volatile float fps = 0.0F;
};

RuntimeStats stats;
httpd_handle_t page_server = nullptr;
httpd_handle_t stream_server = nullptr;
bool access_point_active = false;

constexpr char stream_boundary[] = "frame";
constexpr char stream_content_type[] = "multipart/x-mixed-replace;boundary=frame";
constexpr char stream_part[] = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

void draw_faces(camera_fb_t *frame, const FaceBox *boxes, size_t count) {
  fb_data_t canvas{};
  canvas.width = static_cast<int>(frame->width);
  canvas.height = static_cast<int>(frame->height);
  canvas.data = frame->buf;
  canvas.bytes_per_pixel = 2;
  canvas.format = FB_RGB565;

  constexpr uint32_t yellow_rgb565 = 0xFFE0;
  for (size_t index = 0; index < count; ++index) {
    const int x = boxes[index].x;
    const int y = boxes[index].y;
    const int width = boxes[index].width;
    const int height = boxes[index].height;
    if (width <= 0 || height <= 0) continue;
    fb_gfx_drawFastHLine(&canvas, x, y, width, yellow_rgb565);
    fb_gfx_drawFastHLine(&canvas, x, y + height - 1, width, yellow_rgb565);
    fb_gfx_drawFastVLine(&canvas, x, y, height, yellow_rgb565);
    fb_gfx_drawFastVLine(&canvas, x + width - 1, y, height, yellow_rgb565);
  }
}

esp_err_t index_handler(httpd_req_t *request) {
  httpd_resp_set_type(request, "text/html; charset=utf-8");
  return httpd_resp_send(request, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

esp_err_t status_handler(httpd_req_t *request) {
  char json[224];
  snprintf(json, sizeof(json),
           "{\"faces\":%u,\"candidates\":%u,\"inference_ms\":%u,\"fps\":%.1f,"
           "\"model_trained\":true,\"detector\":\"ESP-DL two-stage\"}",
           stats.faces, stats.candidates, stats.inference_ms, stats.fps);
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_sendstr(request, json);
}

esp_err_t stream_handler(httpd_req_t *request) {
  httpd_resp_set_type(request, stream_content_type);
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");

  FaceDetector detector;
  FaceBox boxes[10];
  int64_t previous_frame_us = esp_timer_get_time();

  while (true) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == nullptr) return ESP_FAIL;

    const int64_t inference_start_us = esp_timer_get_time();
    const size_t face_count = detector.infer(frame->buf, frame->width, frame->height, boxes, 10);
    stats.inference_ms = (esp_timer_get_time() - inference_start_us) / 1000;
    stats.faces = face_count;
    stats.candidates = detector.last_candidate_count();
    draw_faces(frame, boxes, face_count);

    uint8_t *jpeg = nullptr;
    size_t jpeg_length = 0;
    const bool encoded = fmt2jpg(frame->buf, frame->len, frame->width, frame->height,
                                 PIXFORMAT_RGB565, 80, &jpeg, &jpeg_length);
    esp_camera_fb_return(frame);
    if (!encoded) return ESP_FAIL;

    char header[96];
    const int header_length = snprintf(header, sizeof(header), stream_part, jpeg_length);
    esp_err_t result = httpd_resp_send_chunk(request, header, header_length);
    if (result == ESP_OK) result = httpd_resp_send_chunk(request, reinterpret_cast<char *>(jpeg), jpeg_length);
    if (result == ESP_OK) result = httpd_resp_send_chunk(request, "\r\n", 2);
    free(jpeg);
    if (result != ESP_OK) break;

    const int64_t now_us = esp_timer_get_time();
    stats.fps = 1000000.0F / static_cast<float>(now_us - previous_frame_us);
    previous_frame_us = now_us;
  }
  return ESP_OK;
}

bool initialize_camera() {
  camera_config_t config{};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = camera_pins::d0;
  config.pin_d1 = camera_pins::d1;
  config.pin_d2 = camera_pins::d2;
  config.pin_d3 = camera_pins::d3;
  config.pin_d4 = camera_pins::d4;
  config.pin_d5 = camera_pins::d5;
  config.pin_d6 = camera_pins::d6;
  config.pin_d7 = camera_pins::d7;
  config.pin_xclk = camera_pins::xclk;
  config.pin_pclk = camera_pins::pclk;
  config.pin_vsync = camera_pins::vsync;
  config.pin_href = camera_pins::href;
  config.pin_sccb_sda = camera_pins::sda;
  config.pin_sccb_scl = camera_pins::scl;
  config.pin_pwdn = camera_pins::pwdn;
  config.pin_reset = camera_pins::reset;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  return esp_camera_init(&config) == ESP_OK;
}

void start_servers() {
  httpd_config_t page_config = HTTPD_DEFAULT_CONFIG();
  page_config.max_uri_handlers = 4;
  httpd_start(&page_server, &page_config);
  const httpd_uri_t index_uri = {.uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = nullptr};
  const httpd_uri_t status_uri = {.uri = "/api/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = nullptr};
  httpd_register_uri_handler(page_server, &index_uri);
  httpd_register_uri_handler(page_server, &status_uri);

  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = 81;
  stream_config.ctrl_port = 32769;
  stream_config.stack_size = 10240;
  httpd_start(&stream_server, &stream_config);
  const httpd_uri_t stream_uri = {.uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = nullptr};
  httpd_register_uri_handler(stream_server, &stream_uri);
}

void print_browser_link() {
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("Link: http://%s\n", WiFi.localIP().toString().c_str());
  } else if (access_point_active) {
    Serial.printf("Link: http://%s\n", WiFi.softAPIP().toString().c_str());
    Serial.println("Rede Wi-Fi: TinyML-Camera");
  } else {
    Serial.println("Wi-Fi ainda esta conectando.");
  }
}

void process_serial_commands() {
  static String command;
  while (Serial.available() > 0) {
    const char character = static_cast<char>(Serial.read());
    if (character == '\r') continue;
    if (character != '\n') {
      if (command.length() < 32) command += character;
      continue;
    }

    command.trim();
    command.toLowerCase();
    if (command == "ip") {
      print_browser_link();
    } else if (!command.isEmpty()) {
      Serial.println("Comando desconhecido. Digite: ip");
    }
    command = "";
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Conectando ao Wi-Fi");
  const unsigned long wifi_started_at = millis();
  unsigned long last_progress_at = wifi_started_at;
  while (WiFi.status() != WL_CONNECTED && millis() - wifi_started_at < 15000) {
    process_serial_commands();
    if (millis() - last_progress_at >= 500) {
      Serial.print('.');
      last_progress_at = millis();
    }
    delay(10);
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nRede configurada indisponivel. Iniciando ponto de acesso.");
    WiFi.disconnect();
    WiFi.mode(WIFI_AP);
    access_point_active = WiFi.softAP("TinyML-Camera");
    if (!access_point_active) {
      Serial.println("ERRO: falha ao iniciar o ponto de acesso.");
      return;
    }
  } else {
    Serial.println("\nWi-Fi conectado.");
  }

  if (!psramFound()) {
    Serial.println("ERRO: PSRAM nao detectada. Confira a configuracao qio_opi.");
    return;
  }
  if (!initialize_camera()) {
    Serial.println("ERRO: falha ao iniciar a camera.");
    return;
  }
  start_servers();
  Serial.println();
  print_browser_link();
  Serial.println("Digite ip para mostrar o link novamente.");
}

void loop() {
  process_serial_commands();
  delay(10);
}
