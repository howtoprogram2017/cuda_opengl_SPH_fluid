#pragma once
#include"cuda_runtime.h"
#include <vector>

class TriangleMesh
{
public:
	typedef std::vector<unsigned int> Faces;
	typedef std::vector<double3> Normals;
	typedef std::vector<double3> Vertices;

protected:
	Vertices m_x;
	Faces m_indices;
	Normals m_normals;
	Normals m_vertexNormals;

public:
	TriangleMesh();
	~TriangleMesh();

	void release();
	void initMesh(const unsigned int nPoints, const unsigned int nFaces);
	/** Add a new face.	*/
	void addFace(const unsigned int * const indices);
	/** Add a new face.	*/
	void addFace(const int * const indices);
	/** Add new vertex. */
	void addVertex(const double3 &vertex);

	const Faces& getFaces() const { return m_indices; }
	Faces& getFaces() { return m_indices; }
	const Normals& getFaceNormals() const { return m_normals; }
	Normals& getFaceNormals() { return m_normals; }
	const Normals& getVertexNormals() const { return m_vertexNormals; }
	Normals& getVertexNormals() { return m_vertexNormals; }
	const Vertices& getVertices() const { return m_x; }
	Vertices& getVertices() { return m_x; }

	unsigned int numVertices() const { return static_cast<unsigned int>(m_x.size()); }
	unsigned int numFaces() const { return (unsigned int)m_indices.size() / 3; }

	void updateNormals();
	void updateVertexNormals();
};
TriangleMesh::TriangleMesh()
{
}

TriangleMesh::~TriangleMesh()
{
	release();
}

void TriangleMesh::initMesh(const unsigned int nPoints, const unsigned int nFaces)
{
	m_x.reserve(nPoints);
	m_indices.reserve(nFaces * 3);
	m_normals.reserve(nFaces);
	m_vertexNormals.reserve(nPoints);
}

void TriangleMesh::release()
{
	m_indices.clear();
	m_x.clear();
	m_normals.clear();
	m_vertexNormals.clear();
}

void TriangleMesh::addFace(const unsigned int * const indices)
{
	for (unsigned int i = 0u; i < 3; i++)
		m_indices.push_back(indices[i]);
}

void TriangleMesh::addFace(const int * const indices)
{
	for (unsigned int i = 0u; i < 3; i++)
		m_indices.push_back((unsigned int)indices[i]);
}

void TriangleMesh::addVertex(const double3 &vertex)
{
	m_x.push_back(vertex);
}

void TriangleMesh::updateNormals()
{
	m_normals.resize(numFaces());

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numFaces(); i++)
		{
			// Get first three points of face
			double3 a = m_x[m_indices[3 * i]];
			double3 b = m_x[m_indices[3 * i + 1]];
			double3 c = m_x[m_indices[3 * i + 2]];

			// Create normal
			double3 v1 = b - a;
			double3 v2 = c - a;

			m_normals[i] = cross(v1, v2);
			normalize(m_normals[i]);
		}
	}
}

void TriangleMesh::updateVertexNormals()
{
	m_vertexNormals.resize(numVertices());


	for (unsigned int i = 0; i < numVertices(); i++)
	{
		m_vertexNormals[i] = double3{0.0,0.0,0.0};
	}

	for (unsigned int i = 0u; i < numFaces(); i++)
	{
		double3 n = m_normals[i];
		m_vertexNormals[m_indices[3 * i]] += n;
		m_vertexNormals[m_indices[3 * i + 1]] += n;
		m_vertexNormals[m_indices[3 * i + 2]] += n;
	}

	for (unsigned int i = 0; i < numVertices(); i++)
	{
		normalize( m_vertexNormals[i]);
	}
}

