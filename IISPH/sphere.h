#include<vector>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#ifndef  SHPERE_H
#define SPHERE_H
using namespace std;

struct cudaGraphicsResource *cuda_vbo_resource[2];
class sphere{

public:
	sphere(int lod,float radius) {

		for (int oct = 0; oct < 8; oct++) {
			vector<int>* scale = pickOctant(oct);
			vector<int> Scale = *scale;
			bool flip = Scale[0] * Scale[1] * Scale[2]>0;
			for (int i = 0; i <= lod; i++) {
				// Generate a row of vertices on the surface of the sphere
				// using barycentric coordinates.
				int offset = (lod + 1)*(lod + 2)*oct / 2;
				for (int j = 0; i + j <= lod; j++) {
					float a = fix((float)i /(float) lod);
					float b = fix((float)j / (float)lod);
					float c = fix((float)(lod - i - j) / (float)lod);
					float x = a * (float)Scale[0]; //  not unified vector
					float y = b * (float)Scale[1];
					float z = c * (float)Scale[2];
					float square = x * x + y * y + z * z;
					float ratio = radius / (sqrt(square));
					pointData.push_back(x*ratio);
					pointData.push_back(y*ratio);
					pointData.push_back(z*ratio);
				}

				// Generate triangles from this row and the previous row.
				if (i > 0) {
					for (int j = 0; i + j <= lod; j++) {
						int a = (i - 1) * (lod + 1) + ((i - 1) - (i - 1) * (i - 1)) / 2 + j;
						int b = i * (lod + 1) + (i - i * i) / 2 + j;
						
						indices.push_back(a+offset);
						indices.push_back(a+1 + offset);
						indices.push_back(b + offset);

							//tri(data[a], data[a + 1], data[b]));
						if (i + j < lod) {
							//mesh.triangles.push(tri(data[b], data[a + 1], data[b + 1]));
							indices.push_back(b + offset);
							indices.push_back(a + 1 + offset);
							indices.push_back(b +1  + offset);
						}
					}
				}
			}
		}

	};
	void generateBuffer() {
		unsigned int VBO, VAO, EBO, Location[2];
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);
		glGenBuffers(2, &Location[0]);
		//glEnableVertexAttribArray(0);
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*pointData.size(),&(pointData[0]), GL_STATIC_DRAW);

		// position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(0);
		//GLuint Location;
		//glGenBuffers(1, &Location);
		
	
		
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(int)*indices.size(),&(indices[0]),GL_STATIC_DRAW);
		this->VAO = VAO; this->VBO = VBO;
	}
	GLuint getVBO() {
		return VBO;
	}
	GLuint getVAO() { return VAO; }
	int getElementSize() { return indices.size(); };

private:
	vector<int> indices;
	vector<float> pointData;
	GLuint VBO, VAO;
	vector<int>* pickOctant(int i) {
		return new vector<int>{ (i & 1) * 2 - 1, (i & 2) - 1, (i & 4) / 2 - 1 };
	}
	float fix(float x) {
		return x + (x - x * x) / 2;
	}

};

#endif // ! SHPERE_H
