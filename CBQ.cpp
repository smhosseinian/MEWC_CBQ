/*_________________________________________________________________________________________________________
|                                                                                                          |
|  CBQ algorithm (Combinatorial Bnb algorithm with Quadratic relaxation upper bound for the MEWC problem)  |
|                                                                                                          |
|                Copyright (c) 2019 Seyedmohammadhossein Hosseinian. All rights reserved.                  |
|                                                                                                          |
|__________________________________________________________________________________________________________|


 ***  READ ME  ********************************************************************************************

  (1) This code uses Intel(R) Math Kernel Library (MKL), an optimized version of LAPACK/BLAS libraries for 
      implementation on Intel(R) CPUs. 
  (2) Intel(R) MKL library works only with "Intel C++" or "Visual C++" compilers.
  (3) In the following code, (%%%) indicates intermediate output (display) to observe the prunning process.
  (4) In the following code, (^^^) indicates intermediate output (display) to observe comparison of 
      QP relaxation bound and the trivial (TRV) bound of sum of edge weights.
  (5) Before running the code, define N and GRAPH, e.g., 
      #define N 200
      #define GRAPH "brock200_2.clq"

 ***********************************************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <Windows.h>
#include <stdlib.h>
#include <stdio.h>
#include "mkl_lapacke.h"

#define N __
#define GRAPH __
#define LDA N

using namespace std;

#pragma region "Time Record"

double get_wall_time() {
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq)) {
		return 0;
	}
	if (!QueryPerformanceCounter(&time)) {
		return 0;
	}
	return (double)time.QuadPart / freq.QuadPart;
}

double get_cpu_time() {
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
		return
			(double)(d.dwLowDateTime |
			((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	}
	else {
		return 0;
	}
}

#pragma endregion

#pragma region "Heuristic"

struct elm {									//keeps a "index-value" pairs, used for sorting "indices" based on "values"
	int n;
	double val;
};

bool ValueCmp(elm const & a, elm const & b)		//comparison method based on "value" attribute of a "index-value" pair (i.e. elm)
{
	return a.val > b.val;
}

struct clq {									//clique: keeps list of vertices as well as the clique's total edge weight
	vector<int> vertexList;
	double weight;
};

double* makeQ(double **Adj) {					//generates the upper-triangle part of matrix Q, and puts it in a 1-d array to be used in the eigen-decomposition algorithm
	double sumWeight[N];
	for (int i = 0; i<N; i++) {
		sumWeight[i] = 0;
		for (int j = 0; j<N; j++) {
			sumWeight[i] += Adj[i][j];
		}
	}
	double Q[N][N];
	for (int i = 0; i<N; i++) {
		for (int j = 0; j<N; j++) {
			if (j<i) {
				Q[i][j] = 0;
			} else {
				Q[i][j] = Adj[i][j];
			}
		}
	}
	for (int i = 0; i<N - 1; i++) {
		for (int j = i + 1; j<N; j++) {
			if (Q[i][j] == 0) {
				Q[i][j] = -(sumWeight[i]>sumWeight[j] ? sumWeight[i] : sumWeight[j]) - 1;
			}
		}
	}
	double* Q_a=new double[N*N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			Q_a[i*N + j] = Q[i][j];
		}
	}
	return Q_a;
}

clq extractClique(double** Adj, vector<elm> & EVec) {		//extracts clique based on a sorted list of "index-value" pair (will be used to extract clique based on a sorted eigenvector)
	clq clique;
	clique.vertexList.push_back(EVec[0].n);
	clique.weight = 0;
	for (int k = 1; k < N; k++) {
		bool belongs = true;
		double tempW = 0;
		for (int l = 0; l < clique.vertexList.size(); l++) {
			if (Adj[EVec[k].n][clique.vertexList[l]] > 0) {
				tempW += Adj[EVec[k].n][clique.vertexList[l]];
			}
			else {
				belongs = false;
				break;
			}
		}
		if (belongs == true) {
			clique.vertexList.push_back(EVec[k].n);
			clique.weight += tempW;
		}
	}
	return clique;
}

double CCH(double **Adj)		
{
	double* Q_a = makeQ(Adj);
	double lambda[N];
	double BestWeight = 0;
	clq tempClique;
	vector<elm> EVec(N);
	double E[N][N];
	MKL_INT n = N, lda = LDA, info;
	info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, Q_a, lda, lambda);
	if (info > 0) {
		printf("The algorithm failed to compute eigenvalues.\n");
		exit(1);
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			E[i][j] = Q_a[i*N + j];
		}
	}
	for (int j = 0; j<N; j++) {
		for (int i = 0; i<N; i++) {
			EVec[i].n = i;
			EVec[i].val = E[i][j];
		}
		sort(EVec.begin(), EVec.end(), ValueCmp);
		tempClique = extractClique(Adj, EVec);
		if (tempClique.weight>BestWeight) {
			BestWeight = tempClique.weight;
		}
		for (int i = 0; i<N; i++) {
			EVec[i].val *= -1;
		}
		sort(EVec.begin(), EVec.end(), ValueCmp);
		tempClique = extractClique(Adj, EVec);
		if (tempClique.weight>BestWeight) {
			BestWeight = tempClique.weight;
		}
	}
	delete[] Q_a;
	return BestWeight;
}

#pragma endregion

#pragma region "Initial Sort"

struct node {
	int n;
	int degree;			//degree of the vertex in the subgraph induced by R (changing as R is updated)
	int ex_deg;			//sum of "degree" of the vertices adjacent to this vertex in the subgraph induced by R (See definition of ex_deg(q) in Tomita(2007) page 101)
};

bool degCmp(node const & a, node const & b)
{
	return a.degree > b.degree;
}

bool ex_degCmp(node const & a, node const & b)
{
	return a.ex_deg < b.ex_deg;
}

int* sortV(double** Adj, int & Delta) {			//sorts the vertices based on "degree" and "ex_degree" (See definition of ex_deg(q) in Tomita(2007) page 101)
	int* V = new int[N];
	vector<node> R;
	vector<node> Rmin;
	node v;
	int dlt = 0;
	for (int i = 0; i < N; i++) {
		v.n = i;
		v.degree = 0;
		for (int j = 0; j < N; j++) {
			if (Adj[i][j] > 0) {
				v.degree +=1;
			}
		}
		if (v.degree>dlt) {
			dlt = v.degree;
		}
		R.push_back(v);
	}
	Delta = dlt;								//inputs Delta and change its value in the function after calculating the degree of all vertices
	sort(R.begin(), R.end(), degCmp);			//Sorts "node"s in R in a decreasing order "degree"
	int minDeg = (R.end()-1)->degree;
	vector<node>::iterator itr = R.end()-1;
	while(itr->degree == minDeg){
		Rmin.push_back(*itr);
		if (itr == R.begin()) {
			break;
		}
		else {
			itr--;
		}
	}
	node p;										//The "node" with the minimum "ex_deg" among nodes in Rmin
	for (int k = N - 1; k >= 0; k--) {
		if (Rmin.size() == 1) {
			p = Rmin[0];
		} 
		else {
			for (vector<node>::iterator itr_1 = Rmin.begin(); itr_1 != Rmin.end(); itr_1++) {
				itr_1->ex_deg = 0;
				for (vector<node>::iterator itr_2 = R.begin(); itr_2 != R.end(); itr_2++) {
					if (Adj[itr_1->n][itr_2->n] > 0) {
						itr_1->ex_deg += itr_2->degree;
					}
				}
			}
			sort(Rmin.begin(), Rmin.end(), ex_degCmp);				//Sorts "node"s in Rmin in an increasing order "ex_deg"
			p = Rmin[0];
		}
		V[k] = p.n;
		Rmin.clear();
		vector<node>::iterator itr = R.end()-1;
		while (itr != R.begin()) {
			if (itr->n == p.n) {
				itr = R.erase(itr);
				break;
			}
			else {
				itr--;
			}
		}
		for (vector<node>::iterator itr_1 = R.begin(); itr_1 != R.end(); itr_1++) {
			if (Adj[itr_1->n][p.n] > 1) {
				itr_1->degree -= 1;
			}
		}
		sort(R.begin(), R.end(), degCmp);
		minDeg = (R.end() - 1)->degree;
		itr = R.end() - 1;
		while (itr->degree == minDeg) {
			Rmin.push_back(*itr);
			if (itr == R.begin()) {
				break;
			}
			else {
				itr--;
			}
		}
	}
	return V;
}

#pragma endregion

#pragma region "BnB Search"

struct vertex {
	int n;
};

class Equation {
public:										//lam, b_p, q_p are parameters of the eqution
	vector<double> lam;
	vector<double> b_p;
	vector<double> q_p;
	double m;
public:
	Equation (vector<double> _lambda, vector<double> _b_p, vector<double> _q_p, double _m) :lam(_lambda), b_p(_b_p), q_p(_q_p), m(_m) {}

	double evalFunc(double x) {				//evaluates function value at point x
		double f = -m/4;
		double num;
		double denum;
		for (int i = 0; i < m; i++) {
			num = ((lam[i] * b_p[i]) + q_p[i])*((lam[i] * b_p[i]) + q_p[i]);
			denum = (lam[i]-x)*(lam[i] - x);
			f += (num / denum);
		}
		return f;
	}

	double root(double a , double b) {		//finds the root of a monotonic function in the interval [a,b]
		double c = a;
		double fa = evalFunc(a);
		double fb = evalFunc(b);
		double fc = fa;
		do {
			double d1 = b - a;
			if (fabs(fc)<fabs(fb)) {
				a = b;b = c;c = a;
				fa = fb;fb = fc;fc = fa;
			}
			double d2 = (c - b) / 2.0;
			double eps = DBL_EPSILON*(2.0*fabs(b) + 0.5);
			if (fabs(d2) <= eps || !fb)return b;
			if (fabs(d1) >= eps&&fabs(fa)>fabs(fb)) {
				double p, q;
				double cb = c - b;
				double t1 = fb / fa;
				if (a == c) {
					p = cb*t1;
					q = 1.0 - t1;
				}
				else {
					double t2 = fb / fc;
					q = fa / fc;
					p = t1*(cb*q*(q - t2) - (b - a)*(t2 - 1.0));
					q = (q - 1.0)*(t1 - 1.0)*(t2 - 1.0);
				}
				if (p>0.0)q = -q;
				else p = -p;
				if (2.0*p<1.5*cb*q - fabs(eps*q) && 2.0*p<fabs(d1))d2 = p / q;
			}
			if (fabs(d2)<eps)d2 = (d2>0.0 ? eps : -eps);
			a = b;
			fa = fb;
			b += d2;
			fb = evalFunc(b);
			if (fb>0.0&&fc>0.0 || fb<0.0&&fc<0.0) {
				c = a;
				fc = fa;
			}
		} while (true);
	}
};

double* makeQ_L(double** adj , int m , vector<double> q) {			//generates the upper_triangle part of the matrix Q_L, and puts it a 1-d array to be used in the eigen-decomposition algorithm
	double* sumWeight = new double[m];
	for (int i = 0; i<m; i++) {
		sumWeight[i] = 0;
		for (int j = 0; j<m; j++) {
			sumWeight[i] += adj[i][j];
		}
	}
	double** Q_L = new double*[m];
	for (int i = 0; i < m; i++) {
		Q_L[i] = new double[m];
	}
	for (int i = 0; i<m; i++) {
		for (int j = 0; j<m; j++) {
			if (j<i) {
				Q_L[i][j] = 0;
			}
			else {
				Q_L[i][j] = adj[i][j];
			}
		}
	}
	for (int i = 0; i<m - 1; i++) {
		for (int j = i + 1; j<m; j++) {
			if (Q_L[i][j] == 0) {
				Q_L[i][j] = -((sumWeight[i]+q[i]) > (sumWeight[j]+ q[j]) ? (sumWeight[i] + q[i]) : (sumWeight[j] + q[j])) - 1;
			}
		}
	}
	double* Q_La = new double[m*m];
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			Q_La[i*m + j] = Q_L[i][j];
		}
	}
	for (int i = 0; i < m; i++)  delete[] Q_L[i];
	delete[] Q_L;
	delete[] sumWeight;
	return Q_La;
}

bool prune(double ** Adj, const vector<vertex> & candList, const vector<int> & S, double W, double Wmax, int & qp_bound_count, int & trv_bound_count, double & qp_trv, int & badMAT) {
	bool prn = false;
	int d = S.size();
	int m = candList.size();
	vector<double> b(m , 0.5);
	vector<double> q(m , 0);
	double sumLINKweight = 0;
	if (d != 0) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < d; j++) {
				q[i] += Adj[candList[i].n][S[j]];
			}
			sumLINKweight += q[i];
		}
	}
	double sumCANDweight = 0;
	double Z_L = 0;
	if (m==1) {
		Z_L += q[0];
	}
	else {
		double** adj_L = new double*[m];
		for (int i = 0; i < m; i++) {
			adj_L[i] = new double[m];
		}
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				adj_L[i][j] = Adj[candList[i].n][candList[j].n];
				sumCANDweight += 0.5 * Adj[candList[i].n][candList[j].n];
			}
		}
		double* Q_La = makeQ_L(adj_L, m, q);
		double* lambda = new double[m];
		double** E = new double*[m];
		for (int i = 0; i < m; i++) {
			E[i] = new double[m];
		}
		MKL_INT n = m, lda = m, info;
		info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, Q_La, lda, lambda);
		if (info > 0) {
			printf("The algorithm failed to compute eigenvalues.\n");
			exit(1);
		}
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				E[i][j] = Q_La[i*m + j];
			}
		}
		vector<double> _b_p(m, 0);
		vector<double> _q_p(m, 0);
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < m; i++) {
				_b_p[j] += E[i][j] * b[i];
				_q_p[j] += E[i][j] * q[i];
			}
		}
		vector<double> _lambda(m);
		for (int i = 0; i < m; i++) {
			_lambda[i] = lambda[i];
		}
		double lambda_max = *(_lambda.end() - 1);
		int k = m-1;
		while (lambda[k] == lambda_max) {
			if (lambda[k] * _b_p[k] + _q_p[k] != 0) {
				Equation eqa(_lambda, _b_p, _q_p, m);
				double lower_bound = *(_lambda.end() - 1) + 1e-7;
				double upper_bound = 2 * lower_bound;
				while (eqa.evalFunc(upper_bound) > 0.0) upper_bound += lower_bound;
				double mu = eqa.root(lower_bound, upper_bound);
				double num_1;
				double num_2;
				double denum;
				for (int i = 0; i < m; i++) {
					num_1 = (mu * _b_p[i]) + _q_p[i];
					num_2 = (_lambda[i] * ((mu * _b_p[i]) - _q_p[i])) + (2 * mu * _q_p[i]);
					denum = (_lambda[i] - mu) * (_lambda[i] - mu);
					Z_L += ((num_1 * num_2) / denum);
				}
				Z_L = 0.5 * Z_L;
				break;
			}
			k--;
		}
		for (int i = 0; i < m; i++)  delete[] adj_L[i];
		delete[] adj_L;
		delete[] Q_La;
		delete[] lambda;
		for (int i = 0; i < m; i++)  delete[] E[i];
		delete[] E;
	}
	//cout << "  " << W + Z_L;												(%%%)
	//cout << endl << "QP bound:  " << Z_L;									(^^^)
	//cout << '\t' << "TRV bound: " << sumLINKweight + sumCANDweight;		(^^^)
	//cout << '\t' << "BEST:  ";											(^^^)
	double upperBound;
	if (Z_L == 0) {
		badMAT++;
		upperBound = sumLINKweight + sumCANDweight;
	}
	else if (Z_L <= sumLINKweight + sumCANDweight) {
		qp_bound_count++;
		upperBound = Z_L;
		//cout << "QP";														(^^^)
	}
	else {
		trv_bound_count++;
		upperBound = sumLINKweight + sumCANDweight;
		//cout << "TRV";													(^^^)
	}
	qp_trv += Z_L / (sumLINKweight + sumCANDweight);
	if ( W + upperBound <= Wmax) {
		prn = true;
	}
	return prn;
}

void EXPAND (double ** Adj, vector<vertex> & U, vector<int> & S, double & W, double & Wmax, int & count, int & qp_bound_count, int & trv_bound_count, double & qp_trv, int & badMAT) {
	count++;
	//cout << endl << count;			(%%%)
	while (! U.empty() ) {
		vertex v;
		v.n = (U.end() - 1)->n;
		bool prn = prune(Adj, U, S, W, Wmax, qp_bound_count, trv_bound_count, qp_trv, badMAT);
		//cout << "  " <<prn;			(%%%)
		if (prn == false) {
			double addedWeight = 0;
			if (S.size() != 0) {
				for (vector<int>::iterator Sit = S.begin(); Sit != S.end(); Sit++) {
					addedWeight += Adj[*Sit][v.n];
				}
				W += addedWeight;
			}
			else {
				W = 0;
			}
			S.push_back(v.n);
			vector<vertex> Uv;
			for (vector<vertex>::iterator itr_1 = U.begin(); itr_1 != U.end(); itr_1++) {
				if (Adj[itr_1->n][v.n] > 0) {
					Uv.push_back(*itr_1);
				}
			}
			if (!Uv.empty()) {
				EXPAND(Adj, Uv, S, W, Wmax, count, qp_bound_count, trv_bound_count, qp_trv, badMAT);
			}
			else if (W > Wmax) {
				Wmax = W;
			}
			vector<int>::iterator itr = S.erase(S.end()-1);
			W -= addedWeight;
		} 
		else {
			return;
		}
		vector<vertex>::iterator Uit;
		if (U.size() == 1) {
			U.clear();
		}
		else {
			Uit = U.end() - 1;
			while (Uit != U.begin()) {
				if (Uit->n == v.n) {
					Uit = U.erase(Uit);
					break;
				}
				else {
					Uit -= 1;
				}
			}
		}
	}
	return;
}

#pragma endregion

int main()
{
	double A[N][N];
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            A[i][j]=0;
        }
    }
    int s,t;
    ifstream data(GRAPH);
    while (data>>s>>t){
        A[s-1][t-1]=((s+t)%200)+1;
        A[t-1][s-1]=((s+t)%200)+1;
    }
    double* Adj[N];
    for (int i=0; i<N; i++){
        Adj[i]=&A[i][0];
    }
	int count = 0;
	int qp_bound_count = 0;
	int trv_bound_count = 0;
	double qp_trv = 0.0;
	int badMAT = 0;
	vector<int> S;
	double W = 0;
	double wall_1 = get_wall_time();
	double cpu_1 = get_cpu_time();
	/***********************************/
	double InitialSol = CCH(Adj);
	/***********************************/
	double cpu_2 = get_cpu_time();
	double wall_2 = get_wall_time();
	double Wmax = InitialSol;
	int Delta;
	int* V = sortV(Adj, Delta);
	vector<vertex> U;
	vertex u;
	for (int i = 0; i < N; i++) {
		u.n = V[i];
		U.push_back(u);
	}
	/**********************************************************************************/
	EXPAND(Adj, U, S, W, Wmax, count, qp_bound_count, trv_bound_count, qp_trv, badMAT);
	/**********************************************************************************/
	double cpu_3 = get_cpu_time();
	double wall_3 = get_wall_time();
	cout << endl;
	cout << "=====================================================" << endl;
	cout << "         CBQ Algorithm for the MEWC problem          " << endl;
	cout << "=====================================================" << endl;
	cout << endl;
	cout << "INSTANCE :              " << GRAPH << endl;
	cout << endl;
	cout << "Heuristic Sol. :        " << InitialSol << endl;
	cout << "# of BnB nodes :        " << count << endl;
	cout << "Max Weight :            " << Wmax << endl;
	cout << "-----------------------------------------------------" << endl;
	//cout << "Heur. Wall time (ms) :  " << 1000 * (wall_2 - wall_1) << endl;
	//cout << "B&B Wall time (ms) :    " << 1000 * (wall_3 - wall_2) << endl;
	//cout << "Total Wall time (ms) :  " << 1000 * (wall_3 - wall_1) << endl;
	//cout << "-----------------------------------------------------" << endl;
	cout << "Heur. CPU time (ms)  :  " << 1000 * (cpu_2 - cpu_1) << endl;
	cout << "B&B CPU time (ms)  :    " << 1000 * (cpu_3 - cpu_2) << endl;
	cout << "Total CPU time (ms)  :  " << 1000 * (cpu_3 - cpu_1) << endl;
	cout << endl;
	cout << "=====================================================" << endl;
	cout << endl;
	cout << "  ***  Comparison of QP and TRV bounds  ***" << endl;
	cout << endl;
	cout << "# QP bound used :     " << qp_bound_count << endl;
	cout << "# TRV bound used :    " << trv_bound_count + badMAT << endl;
	cout << "Average QP/TRV ratio :  " << qp_trv / (qp_bound_count + trv_bound_count) << endl;
	cout << endl;
	cout << "=====================================================" << endl;
	cout << endl;
	cout << "# BAD matrices  :                           " << badMAT << endl;
	cout << endl;
	cout << endl;
};
