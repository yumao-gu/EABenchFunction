#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <map>

#define PI 3.1415926

using namespace std;
using namespace Eigen;
using namespace cv;


struct particle
{
    Vector2d pos;
    double val;
};
vector<particle> particles;
vector<particle> particles_new;
vector<particle> selected_particles;


int population = 20;
int n_max = 4;    //fix:SelectParent
int max_gernation = 1000;

static default_random_engine e((unsigned)time(0));
static uniform_real_distribution<double> n_uniform(0,1);
static uniform_real_distribution<double> n_uniform_region(-10,10);

double Fitness(Vector2d pos,int FUNC = 1)
{
    if (FUNC == 1)
    {
        double val = 0;
        val = pow(pos.x(),2) + 1 * pow((pos.y() - pow(pos.x(),2)),2);
        return val;
    }

    if (FUNC == 2)
    {
        double val = 0;
        val = (pos.x() * pos.x() - 10 * cos(2 * PI * pos.x()) + 10.0)
              + (pos.y() * pos.y() - 10 * cos(2 * PI * pos.y()) + 10.0);
        return val;
    }
}

void Initialization()
{
    for(int i = 0; i < population; i++)
    {
        particle p;
        p.pos.x() = n_uniform_region(e);
        p.pos.y() = n_uniform_region(e);
        p.val = Fitness(p.pos,1);
        particles.push_back(p);
    }
}

void Mutation()
{
    for(int i = 0;i < particles_new.size();i++)
    {
        double x_mutate_prob = n_uniform(e);
        if(x_mutate_prob < 0.2)
        {
            particles_new[i].pos.x() += 0.05;
        }
        else if(x_mutate_prob < 0.4)
        {
            particles_new[i].pos.x() -= 0.05;
        }

        double y_mutate_prob = n_uniform(e);
        if(y_mutate_prob < 0.2)
        {
            particles_new[i].pos.y() += 0.05;
        }
        else if(y_mutate_prob < 0.4)
        {
            particles_new[i].pos.y() -= 0.05;
        }
    }
}

void FitnessAll(int FUNC = 1)
{
    for(int i = 0;i < particles.size();i++)
    {
        particles[i].val = Fitness(particles[i].pos,FUNC);
        //cout<<"FitnessAll"<<'\t'<<particles[i].pos.x()<<'\t'<<particles[i].pos.y()<<'\t'<<particles[i].val<<'\t'<<Fitness(particles[i].pos,1)<<endl;
    }

//    for(int i = 0;i < particles.size();i++)
//    {
//        cout<<i<<'\t'<<particles[i].pos.x()<<'\t'<<particles[i].pos.y()<<'\t'<<particles[i].val<<'\t'<<Fitness(particles[i].pos,1)<<endl;
//    }
}

particle GetCurrentBest(double min_limit)
{
    particle current_best;
    double best_fitness = 100.0;
    for(int i = 0; i < particles.size(); i++)
    {
        if(particles[i].val < best_fitness && particles[i].val > min_limit + 0.001)
        {
            //cout<<"GetCurrentBest\t" << particles[i].val<<"\tmin_limit\t"<<min_limit<<"\tbest_fitness\t"<<best_fitness<<endl;
            best_fitness = particles[i].val;
            current_best = particles[i];
        }
    }
    return current_best;
}

void CrossOver(particle p1,particle p2)
{
    double method = n_uniform(e);
    particle c1,c2;
    if(method < 0.33)
    {
        //cout<<"A"<<endl;
        c1.pos.x() = floor(p1.pos.x()) + (p2.pos.x() - floor(p2.pos.x()));
        c1.pos.y() = floor(p1.pos.y()) + (p2.pos.y() - floor(p2.pos.y()));
        c2.pos.x() = floor(p2.pos.x()) + (p1.pos.x() - floor(p1.pos.x()));
        c2.pos.y() = floor(p2.pos.y()) + (p1.pos.y() - floor(p1.pos.y()));
    }
    else if(method < 0.66)
    {
        //cout<<"B"<<endl;
        c1.pos = {p1.pos.x(),p2.pos.y()};
        c2.pos = {p2.pos.x(),p1.pos.y()};
    }
    else
    {
        //cout<<"C"<<endl;
        c2.pos = (p1.pos + 2 * p2.pos) / 3.0;
        c1.pos = (p1.pos + c2.pos) / 2.0;
    }
    particles_new.push_back(c1);
    particles_new.push_back(c2);
//    cout<<"CrossOver"<<endl;
//    cout<<c1.x()<<'\t'<<c1.y()<<endl;
//    cout<<c2.x()<<'\t'<<c2.y()<<endl;
}

void Selection()
{
    //cout<<"Selection"<<endl;
    selected_particles.clear();
    particle current_selection;
    current_selection = GetCurrentBest(-100.0);
    selected_particles.push_back(current_selection);
    for(int i = 0; i < n_max - 1; i++)
    {
        double current_best = current_selection.val;
        //cout<<i<<'\t'<<current_best<<endl;
        current_selection = GetCurrentBest(current_best);
        selected_particles.push_back(current_selection);
    }
    //cout<<"END Selection"<<endl;
}

particle SelectParent()
{
    double select_prob = n_uniform(e);
    if(select_prob < 0.4)
    {
        return selected_particles[0];
    }
    else if(select_prob < 0.7)
    {
        return selected_particles[1];
    }
    else if(select_prob < 0.9)
    {
        return selected_particles[2];
    }
    else
    {
        return selected_particles[3];
    }
}

void NewGeneration()
{
    Selection();

//    cout<<"NewGeneration selected_particles"<<endl;
//    int j = 0;
//    for(auto selected_particle : selected_particles)
//    {
//        cout<<j<<'\t'<<selected_particle.pos.x()
//            <<'\t'<<selected_particle.pos.y()<<'\t'<<selected_particle.val<<endl;
//        j++;
//    }
//    cout<<" selected_particles"<<endl;



    particles_new.clear();

    while(particles_new.size() < population - n_max)
    {
        particle p1 = SelectParent();
        particle p2 = SelectParent();
        if(p1.pos != p2.pos)
        {
            CrossOver(p1,p2);
        }
    }

    for(int j = 0; j < n_max; j++)
    {
        particles_new.push_back(selected_particles[j]);
    }
    Mutation();
    //particles_new.push_back(particles[selected_particles[0].first]);

    //cout<<"particles_new size : "<<particles_new.size()<<endl;
    particles.clear();
    for(auto particle : particles_new)
    {
        particles.push_back(particle);
    }
    particles_new.clear();

    FitnessAll(2);


//    for(int i = 0;i < particles.size();i++)
//    {
//        cout<<i<<'\t'<<particles[i].pos.x()<<'\t'<<particles[i].pos.y()<<'\t'<<particles[i].val<<endl;
//    }
//    cout<<"END NewGeneration"<<endl;
}

int main()
{
/*    Initialization();
    int i = 0;
    for(auto particle : particles)
    {
        cout<<i<<'\t'<<particle.x()<<'\t'<<particle.y()<<endl;
        i++;
    }

    FitnessAll(particles,1);
//    i = 0;
//    for(auto val : fitness)
//    {
//        cout<<i<<'\t'<<val<<endl;
//        i++;
//    }

    Selection();
    i = 0;
    for(auto selected_particle : selected_particles)
    {
        cout<<i<<'\t'<<selected_particle.first<<'\t'<<particles[selected_particle.first].x()
        <<'\t'<<particles[selected_particle.first].y()<<endl;
        i++;
    }

    Vector2d tmp1 = SelectParent();
    cout<<tmp1.x()<<'\t'<<tmp1.y()<<endl;
    Vector2d tmp = SelectParent();
    cout<<tmp.x()<<'\t'<<tmp.y()<<endl;

    CrossOver(tmp1,tmp);

    Mutation(particles_new);
    i = 0;
    for(auto particle : particles_new)
    {
        cout<<i<<'\t'<<particle.x()<<'\t'<<particle.y()<<endl;
        i++;
    }*/
    Initialization();
    FitnessAll(2);
    int i = 0;
    while(i < max_gernation)
    {
        //cout<<"/////////////////////////////////////////////////////////"<<endl;
        NewGeneration();
        cout<<i<<"\tcurrent best fitness : "<<GetCurrentBest(-100.0).val<<endl;
        i++;
        //cout<<"END /////////////////////////////////////////////////////"<<endl;
        if(GetCurrentBest(-100.0).val < 1e-4)
        {
            break;
        }
    }

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
