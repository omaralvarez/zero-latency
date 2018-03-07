#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>

//#include <EGL/egl.h>
//#include <GL/glew.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <nvpipe.h>

#define PORT 9080
//#define DEBUG

const uint32_t width = 1920;
const uint32_t height = 1080;

#ifdef DEBUG
void captureFramebufferPPM(GLuint framebuffer, uint32_t width, uint32_t height, const std::string& path)
{
    // For verification...

    size_t numBytes = width * height * 3;
    uint8_t* rgb = new uint8_t[numBytes];

    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb);

    std::ofstream outFile;
    outFile.open(path.c_str(), std::ios::binary);

    outFile << "P6" << "\n"
            << width << " " << height << "\n"
            << "255\n";

    outFile.write((char*) rgb, numBytes);

    delete[] rgb;
}
#endif


int main(int argc, char* argv[])
{
    /*
     * General demo setup: Init EGL and OpenGL context
     */
    /*EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    EGLint major, minor;
    eglInitialize(display, &major, &minor);

    const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };

    EGLint numConfigs;
    EGLConfig config;
    eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);

    const uint32_t width = 1920;
    const uint32_t height = 1080;

    const EGLint pbufferAttribs[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_NONE,
    };

    EGLSurface surface = eglCreatePbufferSurface(display, config, pbufferAttribs);

    eglBindAPI(EGL_OPENGL_API);
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
    eglMakeCurrent(display, surface, surface, context);
    glewInit();*/

    /*
     * Create OpenGL texture for display
     */
    /*const GLchar* clientVertexShader =
            "#version 330\n"
            "void main() {}";

    const GLchar* clientGeometryShader =
            "#version 330 core\n"
            "layout(points) in;"
            "layout(triangle_strip, max_vertices = 4) out;"
            "out vec2 texcoord;"
            "void main() {"
            "gl_Position = vec4( 1.0, 1.0, 0.0, 1.0 ); texcoord = vec2( 1.0, 1.0 ); EmitVertex();"
            "gl_Position = vec4(-1.0, 1.0, 0.0, 1.0 ); texcoord = vec2( 0.0, 1.0 ); EmitVertex();"
            "gl_Position = vec4( 1.0,-1.0, 0.0, 1.0 ); texcoord = vec2( 1.0, 0.0 ); EmitVertex();"
            "gl_Position = vec4(-1.0,-1.0, 0.0, 1.0 ); texcoord = vec2( 0.0, 0.0 ); EmitVertex();"
            "EndPrimitive();"
            "}";

    const GLchar* clientFragmentShader =
            "#version 330\n"
            "uniform sampler2D tex;"
            "in vec2 texcoord;"
            "out vec4 color;"
            "void main() {"
            "	color = texture(tex, texcoord);"
            "}";

    GLuint clientVertexShaderHandle = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(clientVertexShaderHandle, 1, &clientVertexShader, 0);
    glCompileShader(clientVertexShaderHandle);

    GLuint clientGeometryShaderHandle = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(clientGeometryShaderHandle, 1, &clientGeometryShader, 0);
    glCompileShader(clientGeometryShaderHandle);

    GLuint clientFragmentShaderHandle = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(clientFragmentShaderHandle, 1, &clientFragmentShader, 0);
    glCompileShader(clientFragmentShaderHandle);

    GLuint clientFullscreenQuadProgram = glCreateProgram();
    glAttachShader(clientFullscreenQuadProgram, clientVertexShaderHandle);
    glAttachShader(clientFullscreenQuadProgram, clientGeometryShaderHandle);
    glAttachShader(clientFullscreenQuadProgram, clientFragmentShaderHandle);
    glLinkProgram(clientFullscreenQuadProgram);

    GLuint clientFullscreenTextureLocation = glGetUniformLocation(clientFullscreenQuadProgram, "tex");

    GLuint clientFullscreenVAO;
    glGenVertexArrays(1, &clientFullscreenVAO);

    GLuint clientColorTex;
    glGenTextures(1, &clientColorTex);
    glBindTexture(GL_TEXTURE_2D, clientColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0); // must use RGBA(8) here for CUDA-GL interop
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    cudaGraphicsResource_t clientGraphicsResource;
    cudaGraphicsGLRegisterImage(&clientGraphicsResource, clientColorTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);*/

    /*
     * Init decoder
     */
    nvpipe* decoder = nvpipe_create_decoder(NVPIPE_H264_NV);
    if (!decoder)
    {
        std::cerr << "ERROR - Decoder creation failed!!!!!" << std::endl;
        return 1;
    }

    size_t clientDeviceBufferSize = width * height * 4;
    void* clientDeviceBuffer;
    if (cudaMalloc(&clientDeviceBuffer, clientDeviceBufferSize) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory" << std::endl;
        return 1;
    }

    size_t clientReceiveBufferSize = clientDeviceBufferSize; // Reserve enough space for input
    uint8_t* clientReceiveBuffer = new uint8_t[clientReceiveBufferSize];
    memset (clientReceiveBuffer,0,clientReceiveBufferSize);

    struct sockaddr_in address;
    int sock = 0, valread;
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return -1;
    }
  
    memset(&serv_addr, '0', sizeof(serv_addr));
  
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
      
    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0) 
    {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }
  
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        printf("\nConnection Failed \n");
        return -1;
    }

    uint32_t size;
    int len = read(sock , &size, sizeof(uint32_t));
    size = ntohl(size);
    std::cout << "Read len: " << len << "Size: " << size << std::endl;


    size_t nbuffer;
    for (nbuffer = 0; nbuffer < size
                           ; nbuffer = std::max(nbuffer, (size_t)size)) { /* Watch out for buffer overflow */
        len = read(sock , clientReceiveBuffer, size);

        /* FIXME: Error checking */

        nbuffer += len;
    }

    std::cout << "Received: " << nbuffer << "/" << size << std::endl;

    size_t imgsz = width * height * 4;
    uint8_t* rgb = (uint8_t*) malloc(imgsz);
    memset (rgb,0,imgsz);

    /*
     * Decode to OpenGL texture
     */
    size_t numBytes = clientReceiveBufferSize;
    nvp_err_t decodeStatus = nvpipe_decode(decoder, clientReceiveBuffer, size, rgb, width, height, NVPIPE_RGBA);
    if (decodeStatus != NVPIPE_SUCCESS)
    {
        std::cerr << "Decode failed: " << std::string(nvpipe_strerror(decodeStatus)) << std::endl;
        return 1;
    }

    std::cout << "Decode finished!" << std::endl;

    /*cudaGraphicsMapResources(1, &clientGraphicsResource);
    cudaArray_t clientArray;
    cudaGraphicsSubResourceGetMappedArray(&clientArray, clientGraphicsResource, 0, 0);
    cudaMemcpy2DToArray(clientArray, 0, 0, clientDeviceBuffer, width * 4, width * 4, height, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &clientGraphicsResource);*/

    /*
     * Display decoded frame
     */
    /*glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Only for verification

    glUseProgram(clientFullscreenQuadProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, clientColorTex);
    glUniform1i(clientFullscreenTextureLocation, 0);
    glBindVertexArray(clientFullscreenVAO);
    glDrawArrays(GL_POINTS, 0, 1);*/

#ifdef DEBUG
    // Export rendered image for verification
    captureFramebufferPPM(0, width, height, "2-client.ppm");
#endif

    delete[] clientReceiveBuffer;
    cudaFree(clientDeviceBuffer);
    nvpipe_destroy(decoder);

    //eglTerminate(display);

    std::cout << "Hello Client!" << std::endl;

    return 0;
}