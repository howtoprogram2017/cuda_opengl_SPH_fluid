#pragma once
#include "cuda_runtime.h"
#include "math_define.cuh"
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include  <vector>
#include <unordered_map>
#include "TriangleMesh.h"
#include "math_define.cuh"
#include <omp.h>
	/** \brief Tools to handle std::string objects
	*/
class StringTools
{
public:

	static void tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ")
	{
		std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
		std::string::size_type pos = str.find_first_of(delimiters, lastPos);

		while (std::string::npos != pos || std::string::npos != lastPos)
		{
			tokens.push_back(str.substr(lastPos, pos - lastPos));
			lastPos = str.find_first_not_of(delimiters, pos);
			pos = str.find_first_of(delimiters, lastPos);
		}
	}

};


class OBJLoader
{
public:
	using Vec3f = std::array<float, 3>;
	using Vec2f = std::array<float, 2>;
	struct MeshFaceIndices
	{
		int posIndices[3];
		int texIndices[3];
		int normalIndices[3];
	};
	/** This function loads an OBJ file.
	* Only triangulated meshes are supported.
	*/

	static void loadObj(const std::string &filename, std::vector<float3> *x, std::vector<MeshFaceIndices> *faces, std::vector<float3> *normals, std::vector<float2> *texcoords, const float3 &scale)
	{
		//LOG_INFO << "Loading " << filename;

		std::ifstream filestream;
		filestream.open(filename.c_str());
		if (filestream.fail())
		{
			std::cerr << "Failed to open file: " << filename;
			return;
		}

		std::string line_stream;
		bool vt = false;
		bool vn = false;

		std::vector<std::string> pos_buffer;
		std::vector<std::string> f_buffer;

		while (getline(filestream, line_stream))
		{
			std::stringstream str_stream(line_stream);
			std::string type_str;
			str_stream >> type_str;

			if (type_str == "v")
			{
				//Vec3f pos;
				float3 pos;
			//	pos[0];
				pos_buffer.clear();
				std::string parse_str = line_stream.substr(line_stream.find("v") + 1);
				StringTools::tokenize(parse_str, pos_buffer);
				
					pos.x = stof(pos_buffer[0]) * scale.x;
					pos.y= stof(pos_buffer[1]) * scale.y;
					pos.z = stof(pos_buffer[2]) * scale.z;
				
					

				x->push_back(pos);
			}
			else if (type_str == "vt")
			{
				if (texcoords != nullptr)
				{
					//Vec2f tex;
					float2 tex;
					pos_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("vt") + 2);
					StringTools::tokenize(parse_str, pos_buffer);
					tex.x = stof(pos_buffer[0]);
					tex.y = stof(pos_buffer[1]);

					texcoords->push_back(tex);
					vt = true;
				}
			}
			else if (type_str == "vn")
			{
				if (normals != nullptr)
				{
					float3 nor;
					pos_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("vn") + 2);
					StringTools::tokenize(parse_str, pos_buffer);
					/*for (unsigned int i = 0; i < 3; i++)
						nor[i] = stof(pos_buffer[i]);*/
					nor.x = stof(pos_buffer[0]);
					nor.y = stof(pos_buffer[1]);
					nor.z = stof(pos_buffer[2]);


					normals->push_back(nor);
					vn = true;
				}
			}
			else if (type_str == "f")
			{
				MeshFaceIndices faceIndex;
				if (vn && vt)
				{
					f_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
					StringTools::tokenize(parse_str, f_buffer);
					for (int i = 0; i < 3; ++i)
					{
						pos_buffer.clear();
						StringTools::tokenize(f_buffer[i], pos_buffer, "/");
						faceIndex.posIndices[i] = stoi(pos_buffer[0]);
						faceIndex.texIndices[i] = stoi(pos_buffer[1]);
						faceIndex.normalIndices[i] = stoi(pos_buffer[2]);
					}
				}
				else if (vn)
				{
					f_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
					StringTools::tokenize(parse_str, f_buffer);
					for (int i = 0; i < 3; ++i)
					{
						pos_buffer.clear();
						StringTools::tokenize(f_buffer[i], pos_buffer, "/");
						faceIndex.posIndices[i] = stoi(pos_buffer[0]);
						faceIndex.normalIndices[i] = stoi(pos_buffer[1]);
					}
				}
				else if (vt)
				{
					f_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
					StringTools::tokenize(parse_str, f_buffer);
					for (int i = 0; i < 3; ++i)
					{
						pos_buffer.clear();
						StringTools::tokenize(f_buffer[i], pos_buffer, "/");
						faceIndex.posIndices[i] = stoi(pos_buffer[0]);
						faceIndex.texIndices[i] = stoi(pos_buffer[1]);
					}
				}
				else
				{
					f_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
					StringTools::tokenize(parse_str, f_buffer);
					for (int i = 0; i < 3; ++i)
					{
						faceIndex.posIndices[i] = stoi(f_buffer[i]);
					}
				}
				faces->push_back(faceIndex);
			}
		}
		filestream.close();
	}

};
void loadObj(const std::string &filename, TriangleMesh &mesh, const double3 &scale)
{
	std::vector<float3> x;
	std::vector<float3> normals;
	std::vector<OBJLoader:: MeshFaceIndices> faces;
	float3 s = { (float)scale.x, (float)scale.y, (float)scale.z };
	OBJLoader::loadObj(filename, &x, &faces, &normals, nullptr, s);

	//mesh.release();
	const unsigned int nPoints = (unsigned int)x.size();
	const unsigned int nFaces = (unsigned int)faces.size();
	mesh.initMesh(nPoints, nFaces);
	for (unsigned int i = 0; i < nPoints; i++)
	{
		mesh.addVertex(double3{ x[i].x, x[i].y, x[i].z });
	}
	for (unsigned int i = 0; i < nFaces; i++)
	{
		// Reduce the indices by one
		int posIndices[3];
		for (int j = 0; j < 3; j++)
		{
			posIndices[j] = faces[i].posIndices[j] - 1;
		}

		mesh.addFace(&posIndices[0]);
	}


}
class PoissonDiskSampling
{
	typedef uint3 CellPos;

	struct CellPosHasher
	{
		std::size_t operator()(const CellPos& k) const
		{
			const int p1 = 73856093 * k.x;
			const int p2 = 19349663 * k.y;
			const int p3 = 83492791 * k.z;
			return (size_t)(p1 + p2 + p3);
		}
	};
public:
	PoissonDiskSampling();

	/** \brief Struct to store the information of the initial points
	*/
	struct InitialPointInfo
	{
		CellPos cP;
		double3 pos;
		unsigned int ID;
	};

	/** \brief Struct to store the hash entry (spatial hashing)
	*/
	struct HashEntry
	{
		HashEntry() {};
		std::vector<unsigned int> samples;
		unsigned int startIndex;
	};

	 static int floor(const float v)
	{
		return (int)(v + 32768.f) - 32768;			// Shift to get positive values 
	}

	/** Performs the poisson sampling with the
	* respective parameters. Compare
	* http://graphics.cs.umass.edu/pubs/sa_2010.pdf
	*
	* @param mesh mesh data of sampled body
	* @param vertices vertex data of sampled data
	* @param sampledVertices sampled vertices that will be returned
	* @param minRadius minimal distance of sampled vertices
	* @param numTestpointsPerFace # of generated test points per face of body
	* @param distanceNorm 0: euclidean norm, 1: approx geodesic distance
	* @param numTrials # of iterations used to find samples
	*/
	void sampleMesh(const unsigned int numVertices, const double3 *vertices, const unsigned int numFaces, const unsigned int *faces,
		const double minRadius, const unsigned int numTrials,
		unsigned int distanceNorm, std::vector<double3> &samples);

private:
	double m_r;
	unsigned int m_numTrials;
	unsigned int m_numTestpointsPerFace;
	unsigned int m_distanceNorm;
	std::vector<double3> m_faceNormals;
	std::vector<double> m_areas;
	double m_totalArea;

	double m_cellSize;
	double3 m_minVec;
	double3 m_maxVec;

	std::vector<InitialPointInfo> m_initialInfoVec;
	std::vector<std::vector<CellPos>> m_phaseGroups;

	std::default_random_engine m_generator;
	std::uniform_real_distribution<double> m_uniform_distribution1;
	double m_maxArea;

	void computeFaceNormals(const unsigned int numVertices, const double3 *vertices, const unsigned int numFaces, const unsigned int *faces);
	void determineTriangleAreas(const unsigned int numVertices, const double3 *vertices, const unsigned int numFaces, const unsigned int *faces);
	void generateInitialPointSet(const unsigned int numVertices, const double3 *vertices, const unsigned int numFaces, const unsigned int *faces);
	unsigned int getAreaIndex(const std::vector<double>& areas, const double totalArea);
	void parallelUniformSurfaceSampling(std::vector<double3> &samples);

	void quickSort(int left, int right);
	int partition(int left, int right);
	bool compareCellID(CellPos& a, CellPos& b);

	void determineMinX(const unsigned int numVertices, const double3 *vertices);

	bool nbhConflict(const std::unordered_map<CellPos, HashEntry, CellPosHasher>& kvMap, const InitialPointInfo& iPI);
	bool checkCell(const std::unordered_map<CellPos, HashEntry, CellPosHasher>& kvMap, const CellPos& cell, const InitialPointInfo& iPI);


};

inline void PoissonDiskSampling::sampleMesh(const unsigned int numVertices, const double3 * vertices, const unsigned int numFaces, const unsigned int * faces, const double minRadius, const unsigned int numTrials, unsigned int distanceNorm, std::vector<double3>& samples)
{
	{
		m_r = minRadius;
		m_numTrials = numTrials;
		m_distanceNorm = distanceNorm;

		m_cellSize = m_r / sqrt(3.0);

		// Init sampling
		m_maxArea = numeric_limits<double>::min();
		determineMinX(numVertices, vertices);

		determineTriangleAreas(numVertices, vertices, numFaces, faces);

		const double circleArea = M_PI * minRadius * minRadius;
		const unsigned int numInitialPoints = (unsigned int)(40.0 * (m_totalArea / circleArea));
		//cout << "# Initial points: " << numInitialPoints << endl;

		m_initialInfoVec.resize(numInitialPoints);
		m_phaseGroups.resize(27);

		computeFaceNormals(numVertices, vertices, numFaces, faces);

		// Generate initial set of candidate points
		generateInitialPointSet(numVertices, vertices, numFaces, faces);

		// Find minimal coordinates of object

		// Calculate CellIndices
		const double factor = 1.0 / m_cellSize;

#pragma omp parallel for schedule(static)
		for (int i = 0; i < (int)m_initialInfoVec.size(); i++)
		{
			const double3& v = m_initialInfoVec[i].pos;
			const int cellPos1 = PoissonDiskSampling::floor((v.x - m_minVec.x) * factor) + 1;
			const int cellPos2 = PoissonDiskSampling::floor((v.y - m_minVec.y) * factor) + 1;
			const int cellPos3 = PoissonDiskSampling::floor((v.z - m_minVec.z) * factor) + 1;
			m_initialInfoVec[i].cP = make_uint3(cellPos1, cellPos2, cellPos3);
		}

		// Sort Initial points for CellID
		quickSort(0, (int)m_initialInfoVec.size() - 1);

		// PoissonSampling
		parallelUniformSurfaceSampling(samples);

		// release data
		m_initialInfoVec.clear();
		for (int i = 0; i < m_phaseGroups.size(); i++)
		{
			m_phaseGroups[i].clear();
		}
		m_phaseGroups.clear();
	}
}

void PoissonDiskSampling::computeFaceNormals(const unsigned int numVertices, const double3 * vertices, const unsigned int numFaces, const unsigned int * faces)
{
	m_faceNormals.resize(numFaces);

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numFaces; i++)
		{
			// Get first three points of face
			 double3 a = vertices[faces[3 * i]];
			 double3 b = vertices[faces[3 * i + 1]];
			double3 c = vertices[faces[3 * i + 2]];

			// Create normal
			double3 v1 = b-a;
			double3 v2 = c - a;

			m_faceNormals[i] = cross(v1, v2);
			m_faceNormals[i]=normalize(m_faceNormals[i]);
		}
	}
}
inline void PoissonDiskSampling::determineTriangleAreas(const unsigned int numVertices, const double3 * vertices, const unsigned int numFaces, const unsigned int * faces)
{
	m_areas.resize(numFaces);
	double totalArea = 0.0;
	double tmpMaxArea = numeric_limits<double>::min();

//#pragma omp parallel default(shared)
	{
		// Compute area of each triangle
	//	#pragma omp for reduction(+:totalArea) schedule(static) 
		for (int i = 0; i < (int)numFaces; i++)
		{
			 double3 a = vertices[faces[3 * i]];
			 double3 b = vertices[faces[3 * i + 1]];
			 double3 c = vertices[faces[3 * i + 2]];

			const double3 d1 = b - a;
			const double3 d2 = c - a;

			const double area = length(cross(d1,d2)) / 2.0;
			m_areas[i] = area;
			totalArea += area;
			//tmpMaxArea = max(area, tmpMaxArea);

			if (area > tmpMaxArea)
			{
	//#pragma omp critical
				{
					tmpMaxArea =std::max(area, tmpMaxArea);
				}
			}

		}
	}
	m_maxArea = std::max(tmpMaxArea, m_maxArea);
	m_totalArea = totalArea;
	cout << "aera"<<m_totalArea << endl;
}
inline void PoissonDiskSampling::generateInitialPointSet(const unsigned int numVertices, const double3 * vertices, const unsigned int numFaces, const unsigned int * faces)
{
	m_totalArea = 0.0;

#pragma omp parallel default(shared)
	{
		// Generating the surface points
#pragma omp for schedule(static) 
		for (int i = 0; i < (int)m_initialInfoVec.size(); i++)
		{
			// Drawing random barycentric coordinates
			double rn1 = sqrt(m_uniform_distribution1(m_generator));
			double bc1 = 1.0 - rn1;
			double bc2 = m_uniform_distribution1(m_generator)*rn1;
			double bc3 = 1.0 - bc1 - bc2;

			// Triangle selection with probability proportional to area
			const unsigned int randIndex = getAreaIndex(m_areas, m_totalArea);

			// Calculating point coordinates
			double3 v1 = vertices[faces[3 * randIndex]];
			double3 v2 = vertices[faces[3 * randIndex + 1]];
			double3 v3 = vertices[faces[3 * randIndex + 2]];

			m_initialInfoVec[i].pos = bc1 * v1 + bc2 * v2 + bc3 * v3;
			m_initialInfoVec[i].ID = randIndex;
		}
	}
}
inline unsigned int PoissonDiskSampling::getAreaIndex(const std::vector<double>& areas, const double totalArea)
{
	bool notaccepted = true;
	unsigned int index;
	while (notaccepted)
	{
		index = (int)((double)areas.size()*m_uniform_distribution1(m_generator));
		if (m_uniform_distribution1(m_generator)<areas[index] / m_maxArea)
			notaccepted = false;
	}
	return index;
}
inline void PoissonDiskSampling::parallelUniformSurfaceSampling(std::vector<double3>& samples)
{
	// Sort initial points into HashMap storing only the index of the first point of cell
	// and build phase groups
	unordered_map<CellPos, HashEntry, CellPosHasher> hMap(2 * m_initialInfoVec.size());
	//samples.clear();
	//samples.reserve(m_initialInfoVec.size());

	// Already insert first Initial point as start of first cell in hashmap
	{
		const CellPos& cell = m_initialInfoVec[0].cP;
		HashEntry &entry = hMap[cell];
		entry.startIndex = 0;
		entry.samples.reserve(5);
		int index = cell.x % 3 + 3 * (cell.y % 3) + 9 * (cell.z % 3);
		m_phaseGroups[index].push_back(cell);
	}

	for (int i = 1; i < (int)m_initialInfoVec.size(); i++)
	{
		const CellPos& cell = m_initialInfoVec[i].cP;
		if ((uint3)cell != (uint3)m_initialInfoVec[i-1].cP)
		{
			HashEntry &entry = hMap[cell];
			entry.startIndex = i;
			entry.samples.reserve(5);
			int index = cell.x % 3 + 3 * (cell.y % 3) + 9 * (cell.z % 3);
			m_phaseGroups[index].push_back(cell);
		}
	}
	int sum = 0;
	for (auto &vec : m_phaseGroups)
	{
		sum += vec.size();
		//for(uint3 & e:vec)

	}

	// Loop over number of tries to find a sample in a cell
	for (int k = 0; k < (int)m_numTrials; k++)
	{
		// Loop over the 27 cell groups
		for (int pg = 0; pg < m_phaseGroups.size(); pg++)
		{
			const vector<CellPos>& cells = m_phaseGroups[pg];
			// Loop over the cells in each cell group
			#pragma omp parallel for schedule(static)
			for (int i = 0; i < (int)cells.size(); i++)
			{
				const auto entryIt = hMap.find(cells[i]);
				// Check if cell exists
				if (entryIt != hMap.end())
				{
					// Check if max Index is not exceeded
					HashEntry& entry = entryIt->second;
					if (entry.startIndex + k < m_initialInfoVec.size())
					{
						if ((uint3)m_initialInfoVec[entry.startIndex].cP == (uint3)m_initialInfoVec[entry.startIndex + k].cP)
						{
							// choose kth point from cell
							const InitialPointInfo& test = m_initialInfoVec[entry.startIndex + k];
							// Assign sample
							if (!nbhConflict(hMap, test))
							{
								const int index = entry.startIndex + k;
								#pragma omp critical
								{
									entry.samples.push_back(index);
									samples.push_back(m_initialInfoVec[index].pos);
								}
							}
						}
					}
				}
			}
		}
	}
}
inline void PoissonDiskSampling::quickSort(int left, int right)
{
	if (left < right)
	{
		int index = partition(left, right);
		quickSort(left, index - 1);
		quickSort(index, right);
	}
}

inline int PoissonDiskSampling::partition(int left, int right)
{
	int i = left;
	int j = right;
	double3 tmpPos;
	CellPos tmpCell;
	InitialPointInfo tmpInfo;
	CellPos pivot = m_initialInfoVec[left + (right - left) / 2].cP;

	while (i <= j)
	{
		while (compareCellID(m_initialInfoVec[i].cP, pivot))
			i++;

		while (compareCellID(pivot, m_initialInfoVec[j].cP))
			j--;

		if (i <= j)
		{
			tmpInfo = m_initialInfoVec[i];
			m_initialInfoVec[i] = m_initialInfoVec[j];
			m_initialInfoVec[j] = tmpInfo;
			i++;
			j--;
		}
	}
	return i;
}

inline bool PoissonDiskSampling::compareCellID(CellPos & a, CellPos & b)
{
	if (a.x < b.x) return true;
	if (a.x > b.x) return false;
	if (a.y < b.y) return true;
	if (a.y > b.y) return false;
	if (a.z < b.z) return true;
	if (a.z > b.z) return false;

	return false;
}

inline void PoissonDiskSampling::determineMinX(const unsigned int numVertices, const double3 * vertices)
{
	m_minVec = make_double3(numeric_limits<double>::max(), numeric_limits<double>::max(), numeric_limits<double>::max());

	for (int i = 0; i < (int)numVertices; i++)
	{
		const double3& v = vertices[i];
		m_minVec.x = std::min(m_minVec.x, v.x);
		m_minVec.y = std::min(m_minVec.y, v.y);
		m_minVec.z = std::min(m_minVec.z, v.z);
	}
}

inline bool PoissonDiskSampling::nbhConflict(const std::unordered_map<CellPos, HashEntry, CellPosHasher>& hMap, const InitialPointInfo & iPI)
{
	CellPos nbPos = iPI.cP;

	// check neighboring cells inside to outside
	if (checkCell(hMap, nbPos, iPI))
		return true;
	for (int level = 1; level < 3; level++)
	{
		for (int ud = -level; ud < level + 1; ud += 2 * level)
		{
			for (int i = -level + 1; i < level; i++)
			{
				for (int j = -level + 1; j < level; j++)
				{
					nbPos = make_uint3(i, ud, j) + iPI.cP;
					if (checkCell(hMap, nbPos, iPI))
						return true;
				}
			}

			for (int i = -level; i < level + 1; i++)
			{
				for (int j = -level + 1; j < level; j++)
				{
					nbPos = make_uint3(j, i, ud) + iPI.cP;
					if (checkCell(hMap, nbPos, iPI))
						return true;
				}

				for (int j = -level; j < level + 1; j++)
				{
					nbPos = make_uint3(ud, i, j) + iPI.cP;
					if (checkCell(hMap, nbPos, iPI))
						return true;
				}
			}
		}
	}
	return false;
}

inline bool PoissonDiskSampling::checkCell(const std::unordered_map<CellPos, HashEntry, CellPosHasher>& hMap, const CellPos & cell, const InitialPointInfo & iPI)
{
	const auto nbEntryIt = hMap.find(cell);
	if (nbEntryIt != hMap.end())
	{
		const HashEntry& nbEntry = nbEntryIt->second;
		for (unsigned int i = 0; i < nbEntry.samples.size(); i++)
		{
			 InitialPointInfo& info = m_initialInfoVec[nbEntry.samples[i]];
			double dist;
			if (m_distanceNorm == 0 || iPI.ID == info.ID)
			{
				dist = length(iPI.pos - info.pos);
			}
			else if (m_distanceNorm == 1)
			{
				double3 v = normalize(info.pos - iPI.pos);
				double c1 = dot(m_faceNormals[iPI.ID],v);
				double c2 = dot(m_faceNormals[info.ID],v);

				dist = length(iPI.pos - info.pos);
				if (fabs(c1 - c2) > 0.00001f)
					dist *= (asin(c1) - asin(c2)) / (c1 - c2);
				else
					dist /= (sqrt(1.0 - c1 * c1));
			}
			else
			{
				return true;
			}

			if (dist < m_r)
				return true;
		}
	}
	return false;
}
PoissonDiskSampling::PoissonDiskSampling() :
	m_uniform_distribution1(0.0, 1.0)
{
};
