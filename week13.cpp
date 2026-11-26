// =====================================================================
//  13-апта: Жарықтандыру негіздері. Нормаль, Ламберт заңы, ambient + diffuse, нормаль матрицасы.
//  (12-аптаның үстіне: нормаль атрибуты (stride 5 -> 8), normalMatrix, lightPos, lightColor)
//
//  Сурет: assets/container.png
//
//  БАСҚАРУ — камера (10-аптадағыдай)
//    Тінтуір — қарау, дөңгелек — зум, W S A D — жүру, Space / L-Shift — жоғары / төмен
//    L-Ctrl — жүгіру, R — камераны қайтару, P — күйді шығару, I — Invert Y, Tab — тінтуірді босату
//
//  БАСҚАРУ — 13-апта
//    1   — 1-тапсырма: жарық көзі шеңбер бойымен жүреді (қосу/өшіру)
//    2   — 2-тапсырма: max(dot, 0.0) алып тастау (ambient автоматты түрде 0.4 болады)
//    3   — 3-тапсырма: қате нормаль матрицасы mat3(model)  <->  дұрыс transpose(inverse(...))
//    4   — 3-тапсырма: бірінші кубты созу (scale 1, 2.5, 1 + 45° бұру) және тоқтату
//    O/L — 4-тапсырма: ambientStrength көбейту / азайту (0.0 ... 0.5, dt-мен)
//    5   — 5-тапсырма: Фонг (фрагмент)  <->  Гуро (төбелік шейдер)
//    6   — 5-тапсырма: дайын күй: жарық (0, 0, 1.2), кубтар тоқтайды, камера қайтады
//    F   — кубтардың айналуын тоқтату / жалғастыру (скриншотқа ыңғайлы)
//    K   — Қ1: жарық көзін көрсететін кішкентай ақ куб (қосу/өшіру)
//    Z   — Қ2: бағытталған жарық (күн)
//    C   — Қ3: жарық түсі уақытпен өзгереді
//    X   — Қ4: екінші жарық көзі (қызыл + көк)
//    ESC — шығу
//  Режим терезе тақырыбында көрінеді.
// =====================================================================
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef ASSET_DIR
#define ASSET_DIR "assets"
#endif

int gWidth = 800, gHeight = 600;

// ---------- камера (9-10 апта) ----------
const glm::vec3 CAM_START(0.0f, 0.0f, 3.0f);
glm::vec3 camPos   = CAM_START;
glm::vec3 camFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 camUp    = glm::vec3(0.0f, 1.0f, 0.0f);
const float YAW_START = -90.0f, FOV_START = 45.0f;
float yaw = YAW_START, pitch = 0.0f, lastX = 400.0f, lastY = 300.0f, fov = FOV_START;
bool  firstMouse = true, invertY = false, mouseOn = true;
const float sensitivity = 0.1f;

// ---------- 13-апта: күй ----------
float ambientStrength = 0.1f;                 // 4-тапсырма: uniform
bool  orbit      = false;                     // 1-тапсырма
bool  useMax     = true;                      // 2-тапсырма: false = әдейі қате
bool  wrongNormal = false;                    // 3-тапсырма: mat3(model)
bool  squash     = false;                     // 3-тапсырма: бірінші куб созылған
bool  gouraud    = false;                     // 5-тапсырма
bool  freezeCubes = false;
bool  showLamp   = true;                      // Қ1
bool  dirLight   = false;                     // Қ2
bool  colorful   = false;                     // Қ3
bool  twoLights  = false;                     // Қ4
glm::vec3 staticLight(1.2f, 1.0f, 2.0f);      // базалық бөлімдегі жарық орны

void onResize(GLFWwindow*, int w, int h) {
    glViewport(0, 0, w, h);
    gWidth = w;
    gHeight = (h > 0) ? h : 1;
}

void updateFront() {
    glm::vec3 dir;
    dir.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    dir.y = sin(glm::radians(pitch));
    dir.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    camFront = glm::normalize(dir);
}

void mouse_callback(GLFWwindow*, double xpos, double ypos) {
    if (!mouseOn) return;
    if (firstMouse) { lastX = (float)xpos; lastY = (float)ypos; firstMouse = false; }
    float xoffset = (float)xpos - lastX;
    float yoffset = lastY - (float)ypos;
    if (invertY) yoffset = -yoffset;
    lastX = (float)xpos;
    lastY = (float)ypos;
    yaw   += xoffset * sensitivity;
    pitch += yoffset * sensitivity;
    if (pitch >  89.0f) pitch =  89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
    if (yaw >= 360.0f) yaw -= 360.0f;
    if (yaw <= -360.0f) yaw += 360.0f;
    updateFront();
}

void scroll_callback(GLFWwindow*, double, double yoffset) {
    fov -= (float)yoffset;
    if (fov < 1.0f) fov = 1.0f;
    if (fov > 90.0f) fov = 90.0f;
}

bool pressedOnce(GLFWwindow* window, int key) {
    static bool prev[GLFW_KEY_LAST + 1] = {};
    bool now = glfwGetKey(window, key) == GLFW_PRESS;
    bool fired = now && !prev[key];
    prev[key] = now;
    return fired;
}

// ---------- 11-аптадағы loadTexture ----------
unsigned int loadTexture(const char* path) {
    unsigned int id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    stbi_set_flip_vertically_on_load(true);
    int w, h, n;
    unsigned char* data = stbi_load(path, &w, &h, &n, 0);
    if (data) {
        GLenum fmt = (n == 4) ? GL_RGBA : GL_RGB;
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, fmt, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        std::cout << path << " " << w << "x" << h << " арна: " << n << std::endl;
    } else {
        std::cerr << "Сурет жүктелмеді: " << path << std::endl;
    }
    stbi_image_free(data);
    return id;
}

// ---------- шейдерлер ----------
// 3-қадам: төбелік шейдер (Фонг: жарық фрагментте есептеледі)
const char* vsPhong = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;
out vec2 vTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

void main() {
    FragPos   = vec3(model * vec4(aPos, 1.0));     // ДҮНИЕЛІК координата (view/projection емес!)
    Normal    = normalMatrix * aNormal;
    vTexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

// 4-қадам: фрагмент шейдері (ambient + diffuse). Қосымша режимдер uniform-мен қосылады.
const char* fsPhong = R"(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 vTexCoord;

out vec4 FragColor;

uniform sampler2D ourTexture;
uniform vec3  lightPos;
uniform vec3  lightColor;
uniform float ambientStrength;     // 4-тапсырма
uniform int   useMax;              // 2-тапсырма
uniform int   dirLight;            // Қ2
uniform vec3  lightDirection;      // Қ2
uniform int   useLight2;           // Қ4
uniform vec3  lightPos2;
uniform vec3  lightColor2;

float lambert(vec3 norm, vec3 lightDir) {
    float d = dot(norm, lightDir);
    return (useMax == 1) ? max(d, 0.0) : d;
}

void main() {
    vec3 norm = normalize(Normal);                 // интерполяция нормальды қысқартады
    vec3 lightDir = dirLight == 1 ? normalize(-lightDirection) : normalize(lightPos - FragPos);

    vec3 ambient = ambientStrength * lightColor;
    vec3 diffuse = lambert(norm, lightDir) * lightColor;
    if (useLight2 == 1)                            // Қ4: екінші көздің үлесі қосылады
        diffuse += lambert(norm, normalize(lightPos2 - FragPos)) * lightColor2;

    FragColor = vec4(ambient + diffuse, 1.0) * texture(ourTexture, vTexCoord);
}
)";

// 5-тапсырма: Гуро — жарық төбелік шейдерде есептеледі, фрагмент тек көбейтеді
const char* vsGouraud = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal;

out vec3 LightResult;
out vec2 vTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;
uniform vec3  lightPos;
uniform vec3  lightColor;
uniform float ambientStrength;
uniform int   useMax;

void main() {
    vec3 FragPos  = vec3(model * vec4(aPos, 1.0));
    vec3 norm     = normalize(normalMatrix * aNormal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float d = dot(norm, lightDir);
    if (useMax == 1) d = max(d, 0.0);
    LightResult = ambientStrength * lightColor + d * lightColor;
    vTexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";
const char* fsGouraud = R"(
#version 330 core
in vec3 LightResult;
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D ourTexture;
void main() { FragColor = vec4(LightResult, 1.0) * texture(ourTexture, vTexCoord); }
)";

// Қ1: жарық көзін көрсететін қарапайым шейдер
const char* vsLamp = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() { gl_Position = projection * view * model * vec4(aPos, 1.0); }
)";
const char* fsLamp = R"(
#version 330 core
out vec4 FragColor;
void main() { FragColor = vec4(1.0); }
)";

unsigned int makeShader(const char* vsSrc, const char* fsSrc) {
    auto compile = [](GLenum type, const char* src) {
        unsigned int id = glCreateShader(type);
        glShaderSource(id, 1, &src, nullptr);
        glCompileShader(id);
        int ok;
        glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[512]; glGetShaderInfoLog(id, 512, nullptr, log); std::cerr << log << "\n"; }
        return id;
    };
    unsigned int vs = compile(GL_VERTEX_SHADER, vsSrc), fs = compile(GL_FRAGMENT_SHADER, fsSrc);
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ---------- 1-қадам: vertices[] = позиция(3) + UV(2) + нормаль(3), 36 вершина ----------
// Нормаль — бет қарайтын БІРЛІК вектор: (0,0,1) алдыңғы, (0,0,-1) артқы, (1,0,0) оң, (-1,0,0) сол, (0,1,0) жоғарғы, (0,-1,0) төменгі.
// Бір жақтың алты төбесінде де нормаль бірдей (flat shading).
std::vector<float> buildVertices() {
    struct Face { glm::vec3 n, u, v; };
    const Face faces[6] = {
        {{ 1, 0, 0}, {0, 0,-1}, {0, 1, 0}}, {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
        {{ 0, 1, 0}, {1, 0, 0}, {0, 0,-1}}, {{ 0,-1, 0}, {1, 0, 0}, {0, 0, 1}},
        {{ 0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, {{ 0, 0,-1}, {-1,0, 0}, {0, 1, 0}},
    };
    const float cu[4] = { -0.5f, 0.5f, 0.5f, -0.5f }, cv[4] = { -0.5f, -0.5f, 0.5f, 0.5f };
    const float uvu[4] = { 0, 1, 1, 0 }, uvv[4] = { 0, 0, 1, 1 };
    const int order[6] = { 0, 1, 2, 0, 2, 3 };
    std::vector<float> out;
    for (const Face& f : faces)
        for (int k : order) {
            glm::vec3 p = f.n * 0.5f + f.u * cu[k] + f.v * cv[k];
            out.insert(out.end(), { p.x, p.y, p.z, uvu[k], uvv[k], f.n.x, f.n.y, f.n.z });
        }
    return out;                                      // 36 x 8 = 288 сан
}

int main() {
    if (!glfwInit()) { std::cerr << "GLFW iske qosylmady\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    GLFWwindow* window = glfwCreateWindow(gWidth, gHeight, "Week 13 - lighting", nullptr, nullptr);
    if (!window) { std::cerr << "Tereze zhasalmady\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, onResize);
    glfwSwapInterval(1);
    if (gladLoadGL(glfwGetProcAddress) == 0) { std::cerr << "GLAD zhuktelmedi\n"; return -1; }
    std::cout << "GPU: " << glGetString(GL_RENDERER) << std::endl;

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    updateFront();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    // ---------- VAO / VBO ----------
    std::vector<float> verts = buildVertices();
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
    // 2-қадам: stride ҮШ жерде де 8
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);                    // ұмытсаң — куб қара болады
    glBindVertexArray(0);

    unsigned int phongProg   = makeShader(vsPhong, fsPhong);
    unsigned int gouraudProg = makeShader(vsGouraud, fsGouraud);
    unsigned int lampProg    = makeShader(vsLamp, fsLamp);

    std::string path = std::string(ASSET_DIR) + "/container.png";
    unsigned int texture = loadTexture(path.c_str());

    glm::vec3 cubePositions[] = {
        {0.0f, 0.0f, 0.0f}, {2.0f, 0.5f, -2.0f}, {-1.8f, -0.3f, -1.0f}, {1.2f, -0.8f, -3.5f}, {-2.2f, 1.0f, -4.0f}
    };

    float cubeAngle = 0.0f;                          // freeze үшін өз сағаты
    std::string lastTitle;
    float last = (float)glfwGetTime();
    float savedAmbient = 0.1f;
    while (!glfwWindowShouldClose(window)) {
        float now = (float)glfwGetTime(), dt = std::min(now - last, 0.05f);
        last = now;
        if (!freezeCubes) cubeAngle += dt;

        // ---------- енгізу ----------
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
        float speed = 2.5f * dt;
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) speed *= 2.0f;
        glm::vec3 right = glm::normalize(glm::cross(camFront, camUp));
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camPos += speed * camFront;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camPos -= speed * camFront;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camPos -= right * speed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camPos += right * speed;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camPos += speed * camUp;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camPos -= speed * camUp;
        if (glm::length(camPos) > 20.0f) camPos = glm::normalize(camPos) * 20.0f;

        // 4-тапсырма: ambient — үздіксіз әрекет, сондықтан dt-ге көбейтіледі
        if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) ambientStrength += 0.3f * dt;
        if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) ambientStrength -= 0.3f * dt;
        if (ambientStrength < 0.0f) ambientStrength = 0.0f;
        if (ambientStrength > 0.5f) ambientStrength = 0.5f;

        if (pressedOnce(window, GLFW_KEY_1)) orbit = !orbit;
        if (pressedOnce(window, GLFW_KEY_2)) {
            useMax = !useMax;
            if (!useMax) { savedAmbient = ambientStrength; ambientStrength = 0.4f; }   // айырмашылық көрінсін
            else ambientStrength = savedAmbient;
        }
        if (pressedOnce(window, GLFW_KEY_3)) wrongNormal = !wrongNormal;
        if (pressedOnce(window, GLFW_KEY_4)) { squash = !squash; if (squash) freezeCubes = true; }
        if (pressedOnce(window, GLFW_KEY_5)) gouraud = !gouraud;
        if (pressedOnce(window, GLFW_KEY_6)) {           // 5-тапсырма: дайын күй
            staticLight = glm::vec3(0.0f, 0.0f, 1.2f);
            orbit = false; freezeCubes = true; cubeAngle = 0.0f; squash = false;
            camPos = CAM_START; yaw = YAW_START; pitch = 0.0f; fov = FOV_START; updateFront();
        }
        if (pressedOnce(window, GLFW_KEY_F)) freezeCubes = !freezeCubes;
        if (pressedOnce(window, GLFW_KEY_K)) showLamp = !showLamp;
        if (pressedOnce(window, GLFW_KEY_Z)) dirLight = !dirLight;
        if (pressedOnce(window, GLFW_KEY_C)) colorful = !colorful;
        if (pressedOnce(window, GLFW_KEY_X)) twoLights = !twoLights;
        if (pressedOnce(window, GLFW_KEY_I)) invertY = !invertY;
        if (pressedOnce(window, GLFW_KEY_R)) {
            camPos = CAM_START; yaw = YAW_START; pitch = 0.0f; fov = FOV_START; updateFront();
        }
        if (pressedOnce(window, GLFW_KEY_P)) {
            std::cout << std::fixed << std::setprecision(2) << "camPos = (" << camPos.x << ", " << camPos.y << ", " << camPos.z
                      << ")  ambient = " << ambientStrength << std::endl;
        }
        if (pressedOnce(window, GLFW_KEY_TAB)) {
            mouseOn = !mouseOn;
            glfwSetInputMode(window, GLFW_CURSOR, mouseOn ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
            if (mouseOn) firstMouse = true;
        }

        // ---------- жарық параметрлері (әр кадр: lightPos өзгеруі мүмкін) ----------
        glm::vec3 lightPos = orbit ? glm::vec3(1.5f * std::sin(now), 1.0f, 1.5f * std::cos(now)) : staticLight;
        glm::vec3 lightColor(1.0f);
        if (colorful)                                                       // Қ3
            lightColor = glm::vec3(std::sin(now * 2.0f) * 0.5f + 0.5f, std::sin(now * 0.7f) * 0.5f + 0.5f,
                                   std::sin(now * 1.3f) * 0.5f + 0.5f);
        glm::vec3 lightPos2(-1.5f, -0.5f, 1.5f), lightColor2(0.2f, 0.3f, 1.0f);
        if (twoLights) lightColor = glm::vec3(1.0f, 0.2f, 0.2f);            // Қ4: қызыл + көк
        glm::vec3 lightDirection(-0.2f, -1.0f, -0.3f);                      // Қ2

        // ---------- сызу ----------
        glClearColor(0.07f, 0.08f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 view = glm::lookAt(camPos, camPos + camFront, camUp);
        glm::mat4 projection = glm::perspective(glm::radians(fov), (float)gWidth / (float)gHeight, 0.1f, 100.0f);

        unsigned int shader = gouraud ? gouraudProg : phongProg;
        glUseProgram(shader);
        auto L = [&](const char* name) { return glGetUniformLocation(shader, name); };   // -1 болса glUniform үнсіз өтеді
        glUniformMatrix4fv(L("projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(L("view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniform1i(L("ourTexture"), 0);
        glUniform3fv(L("lightPos"), 1, glm::value_ptr(lightPos));           // 5-қадам; циклдің ІШІНДЕ, себебі lightPos қозғалады
        glUniform3fv(L("lightColor"), 1, glm::value_ptr(lightColor));
        glUniform1f(L("ambientStrength"), ambientStrength);
        glUniform1i(L("useMax"), useMax ? 1 : 0);
        glUniform1i(L("dirLight"), dirLight ? 1 : 0);
        glUniform3fv(L("lightDirection"), 1, glm::value_ptr(lightDirection));
        glUniform1i(L("useLight2"), twoLights ? 1 : 0);
        glUniform3fv(L("lightPos2"), 1, glm::value_ptr(lightPos2));
        glUniform3fv(L("lightColor2"), 1, glm::value_ptr(lightColor2));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindVertexArray(VAO);
        for (int i = 0; i < 5; i++) {
            glm::mat4 model = glm::mat4(1.0f);
            if (i == 0 && squash) {                      // 3-тапсырма: алдымен бұру, содан кейін созу (GLM төменнен жоғары оқиды)
                model = glm::scale(model, glm::vec3(1.0f, 2.5f, 1.0f));
                model = glm::rotate(model, glm::radians(45.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            } else {
                model = glm::translate(model, cubePositions[i]);
                model = glm::rotate(model, cubeAngle * (0.5f + 0.3f * i), glm::vec3(0.3f, 1.0f, 0.0f));
            }
            glUniformMatrix4fv(L("model"), 1, GL_FALSE, glm::value_ptr(model));

            // 5-қадам: normalMatrix әр кубқа БӨЛЕК (әрқайсысының өз model-і бар), mat3 — жылжыту керек емес
            glm::mat3 normalMatrix = wrongNormal ? glm::mat3(model)                                // А нұсқасы — қате
                                                 : glm::transpose(glm::inverse(glm::mat3(model))); // Б нұсқасы — дұрыс
            glUniformMatrix3fv(L("normalMatrix"), 1, GL_FALSE, glm::value_ptr(normalMatrix));
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        if (showLamp && !dirLight) {                     // Қ1: жарық көзінің орнына кішкентай ақ куб
            glUseProgram(lampProg);
            glUniformMatrix4fv(glGetUniformLocation(lampProg, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
            glUniformMatrix4fv(glGetUniformLocation(lampProg, "view"), 1, GL_FALSE, glm::value_ptr(view));
            glm::mat4 lampModel = glm::translate(glm::mat4(1.0f), lightPos);
            lampModel = glm::scale(lampModel, glm::vec3(0.1f));
            glUniformMatrix4fv(glGetUniformLocation(lampProg, "model"), 1, GL_FALSE, glm::value_ptr(lampModel));
            glDrawArrays(GL_TRIANGLES, 0, 36);
            if (twoLights) {
                lampModel = glm::scale(glm::translate(glm::mat4(1.0f), lightPos2), glm::vec3(0.1f));
                glUniformMatrix4fv(glGetUniformLocation(lampProg, "model"), 1, GL_FALSE, glm::value_ptr(lampModel));
                glDrawArrays(GL_TRIANGLES, 0, 36);
            }
        }

        char buf[256];
        snprintf(buf, sizeof(buf), "%s | ambient %.2f | max:%s | normal:%s%s%s%s%s%s", gouraud ? "GOURAUD" : "PHONG",
                 ambientStrength, useMax ? "on" : "OFF", wrongNormal ? "WRONG mat3(model)" : "transpose(inverse)",
                 orbit ? " | orbit" : "", squash ? " | squash" : "", dirLight ? " | directional" : "",
                 colorful ? " | colorful" : "", twoLights ? " | 2 lights" : "");
        if (lastTitle != buf) { glfwSetWindowTitle(window, buf); lastTitle = buf; }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
