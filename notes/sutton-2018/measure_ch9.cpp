// Own seeded experiments for Sutton & Barto (2018), Figs 9.1, 9.2, 9.5, 9.10.
// g++ -O3 -std=c++17 -fopenmp measure_ch9.cpp -o /tmp/ch9
// OMP_NUM_THREADS=4 /tmp/ch9 reference.tsv > measurements_ch9_walk.json
// reference.tsv: 1000 rows, independently solved true value and normalized occupancy.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;
struct RNG {uint32_t a;double next(){uint32_t t=(a+=0x6d2b79f5u);t=(t^(t>>15))*(t|1u);t^=t+(t^(t>>7))*(t|61u);return double(t^(t>>14))/4294967296.;}};
struct Episode {vector<int>s;double G;};
Episode episode(RNG&r){Episode e;int s=500;while(s>=1&&s<=1000){e.s.push_back(s-1);int k=1+int(100*r.next());s+=r.next()<.5?k:-k;}e.G=s<1?-1:1;return e;}
void arr(const vector<double>&v){cout<<'[';for(size_t i=0;i<v.size();i++){if(i)cout<<',';cout<<v[i];}cout<<']';}
void moments(const vector<vector<double>>&r){size_t n=r.size(),k=r[0].size();vector<double>m(k),se(k);for(size_t j=0;j<k;j++){for(auto&v:r)m[j]+=v[j]/n;for(auto&v:r)se[j]+=(v[j]-m[j])*(v[j]-m[j]);se[j]=sqrt(se[j]/(n*(n-1)));}cout<<"{\"mean\":";arr(m);cout<<",\"se\":";arr(se);cout<<'}';}
double err(const vector<double>&w,int K,const vector<double>&V){double e=0;for(int s=0;s<1000;s++){double z=w[s*K/1000]-V[s];e+=z*z;}return sqrt(e/1000);}
int main(int argc,char**argv){if(argc!=2)return 2;cout<<setprecision(14);vector<double>V(1000),mu(1000);ifstream f(argv[1]);for(int s=0;s<1000;s++)if(!(f>>V[s]>>mu[s]))return 3;
 // Fig 9.1: one actual 100000-episode gradient-MC run, every visit.
 RNG r{900001};vector<double>mc(10),mc10;long long mcsteps=0;for(int e=0;e<100000;e++){auto ep=episode(r);mcsteps+=ep.s.size();for(int s:ep.s)mc[s/100]+=.00002*(ep.G-mc[s/100]);if(e==9)mc10=mc;}
 // Fig 9.2 right: paired paths across all alpha,n cases, 100 independent runs.
 vector<vector<double>> nr(100,vector<double>(210));long long nsteps=0;
 #pragma omp parallel for reduction(+:nsteps) schedule(dynamic)
 for(int run=0;run<100;run++){RNG R{uint32_t(910001+run)};vector<Episode>eps;for(int e=0;e<10;e++){eps.push_back(episode(R));nsteps+=eps.back().s.size();}for(int ni=0;ni<10;ni++)for(int ai=0;ai<=20;ai++){int n=1<<ni;double alpha=ai/20.,total=0;vector<double>w(20);for(auto&ep:eps){int T=ep.s.size();for(int t=0;t<T;t++){int stop=min(T,t+n);double G=stop==T?ep.G:w[ep.s[stop]/50];int g=ep.s[t]/50;w[g]+=alpha*(G-w[g]);}total+=err(w,20,V);}nr[run][ni*21+ai]=total/10;}}
 // Figs 9.5 and 9.10: six dense bases and two tile representations.
 // All eight learners see exactly the same 5000 complete episodes per run.
 // Fourier uses caption's constant per-feature alpha (no frequency normalization).
 vector<int>orders={5,10,20,5,10,20};vector<vector<vector<double>>>phi(6);vector<vector<double>>C(6),b(6);double vv=0;for(int s=0;s<1000;s++)vv+=mu[s]*V[s]*V[s];
 for(int m=0;m<6;m++){int d=orders[m]+1;phi[m].assign(1000,vector<double>(d));C[m].assign(d*d,0);b[m].assign(d,0);for(int s=0;s<1000;s++){double x=(s+1)/1000.;for(int j=0;j<d;j++)phi[m][s][j]=m<3?pow(x,j):cos(acos(-1.)*j*x);for(int j=0;j<d;j++){b[m][j]+=mu[s]*V[s]*phi[m][s][j];for(int k=0;k<d;k++)C[m][j*d+k]+=mu[s]*phi[m][s][j]*phi[m][s][k];}}}
 const int dt=350;vector<array<int,50>>tile(1000);for(int s=0;s<1000;s++)for(int t=0;t<50;t++)tile[s][t]=t*7+(s+4*t)/200;
 vector<vector<vector<double>>>curves(8,vector<vector<double>>(30));vector<long long>steps(30);
 #pragma omp parallel for schedule(dynamic)
 for(int run=0;run<30;run++){RNG R{uint32_t(920001+run)};vector<vector<double>>w(6);for(int m=0;m<6;m++)w[m].assign(orders[m]+1,0);vector<double>one(5),many(dt);for(int m=0;m<8;m++)curves[m][run].push_back(sqrt(vv));for(int epno=1;epno<=5000;epno++){auto ep=episode(R);steps[run]+=ep.s.size();for(int s:ep.s){for(int m=0;m<6;m++){auto&x=phi[m][s];double pred=inner_product(x.begin(),x.end(),w[m].begin(),0.);double d=(m<3?.0001:.00005)*(ep.G-pred);for(int j=0;j<=orders[m];j++)w[m][j]+=d*x[j];}one[s/200]+=.0001*(ep.G-one[s/200]);double p=0;for(int t:tile[s])p+=many[t];double d=.0001/50*(ep.G-p);for(int t:tile[s])many[t]+=d;}
 if(epno%25==0){for(int m=0;m<6;m++){int d=orders[m]+1;double e=vv;for(int j=0;j<d;j++){e-=2*w[m][j]*b[m][j];for(int k=0;k<d;k++)e+=w[m][j]*w[m][k]*C[m][j*d+k];}curves[m][run].push_back(sqrt(max(0.,e)));}double e1=0,e50=0;for(int s=0;s<1000;s++){double p=0;for(int t:tile[s])p+=many[t];e1+=mu[s]*pow(V[s]-one[s/200],2);e50+=mu[s]*pow(V[s]-p,2);}curves[6][run].push_back(sqrt(e1));curves[7][run].push_back(sqrt(e50));}}}
 cout<<"{\"rng\":\"Mulberry32; jump then sign\",\"mc\":{\"seed\":900001,\"episodes\":100000,\"steps\":"<<mcsteps<<",\"alpha\":0.00002,\"weights\":";arr(mc);cout<<"},\"nstep\":{\"seeds\":[910001,910100],\"runs\":100,\"episodes\":10,\"distinct_path_steps\":"<<nsteps<<",\"alpha\":[";for(int a=0;a<=20;a++){if(a)cout<<',';cout<<a/20.;}cout<<"],\"n\":[1,2,4,8,16,32,64,128,256,512],\"metric\":\"mean over first10 episodes of unweighted1000-state RMSE\",\"results\":";moments(nr);cout<<"},\"bases\":{\"seeds\":[920001,920030],\"runs\":30,\"episodes\":5000,\"distinct_path_steps\":"<<accumulate(steps.begin(),steps.end(),0LL)<<",\"checkpoints\":[";for(int e=0;e<=5000;e+=25){if(e)cout<<',';cout<<e;}cout<<"],\"metric\":\"on-policy-mu-weighted RMSE, averaged across runs\",\"curves\":[";vector<string>names={"polinom5","polinom10","polinom20","Fourier5","Fourier10","Fourier20","karo1","karo50"};for(int m=0;m<8;m++){if(m)cout<<',';cout<<"{\"name\":\""<<names[m]<<"\",\"results\":";moments(curves[m]);cout<<'}';}cout<<"]},\"fixtures\":{\"mc_first10\":";arr(mc10);cout<<",\"nstep_run0\":";arr(nr[0]);cout<<",\"bases_run0_after25\":[";for(int m=0;m<8;m++){if(m)cout<<',';cout<<curves[m][0][1];}cout<<"]}}\n";
}
