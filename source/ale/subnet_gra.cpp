#include "../stdafx.h"
#include <FL/Fl.H>
#include <FL/gl.h>
#include <FL/glu.h>
#include <FL/Fl_Window.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Round_Button.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Sys_Menu_Bar.H>
#include <FL/Fl_Menu_Item.H>
#include <FL/Fl_Choice.H>
#include "../dati_local.h"
#include "../analisi_dati.h"
#include <physycom/string.hpp>
#include "subnet_gra.h"
#include "../form.h"
#include "../draw.h"
#include "../polyline.h"
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

extern Fl_Window *form;
extern Fl_Check_Button *button32;
extern Fl_Sys_Menu_Bar *menu;
extern bool mostraSubnet, reDraw;
extern int w_est, h_est, space, b_w, b_h, offset_h, r_offset_h;
extern Fl_Output *linea2;
extern string subnet_select;
extern vector<PolyLine> polyline;
extern vector<pair<int, double>> nl_poly;
map<string, vector<int>> subnets;
vector<Fl_Check_Button*> subnet_buttons;
string subnet_label;

// Utils ---------------------------------------------------------------------------------------------
template<typename Container, typename Iterator>
Iterator SafeErase(Container &c, Iterator it)
{
  if (it == c.end()) return it;
  return c.erase(it);
}
//----------------------------------------------------------------------------------------------------
template<typename T>
void HackeraSubnet(T &t)
{
#if defined (CARNEVALE_26) || defined(CARNEVALE_27)
  t.push_back(6144);
  t.push_back(3135);
  t.push_back(2920);
  t.push_back(4115);
  t.push_back(5097);
  t.push_back(5100);
  t.push_back(5099);
  t.push_back(5098);
  t.push_back(6194);
  t.push_back(6197);
  t.push_back(6219);
  t.push_back(5096);
  t.push_back(5095);
  t.push_back(6198);
  t.push_back(6240);
  t.push_back(6239);
  t.push_back(5177);
  t.push_back(5394);
  t.push_back(6177);
  t.push_back(5393);
  t.push_back(6168);
  t.push_back(6171);
  t.push_back(3187);
  t.push_back(6133);
  t.push_back(5508);
  t.push_back(5502);
  t.push_back(6137);
  t.push_back(5506);
  t.push_back(5505);
  t.push_back(5496);
  t.push_back(5495);
  t.push_back(5390);
  t.push_back(5075);

  SafeErase(t, find(t.begin(), t.end(), 2420));
  SafeErase(t, find(t.begin(), t.end(), 2545));
  SafeErase(t, find(t.begin(), t.end(), 5412));
  SafeErase(t, find(t.begin(), t.end(), 4877));
  SafeErase(t, find(t.begin(), t.end(), 5799));
  SafeErase(t, find(t.begin(), t.end(), 6436));
  SafeErase(t, find(t.begin(), t.end(), 5364));
  SafeErase(t, find(t.begin(), t.end(), 1078));
  SafeErase(t, find(t.begin(), t.end(), 948));
  SafeErase(t, find(t.begin(), t.end(), 1602));
  SafeErase(t, find(t.begin(), t.end(), 3749));
  SafeErase(t, find(t.begin(), t.end(), 3750));
  SafeErase(t, find(t.begin(), t.end(), 3753));
  SafeErase(t, find(t.begin(), t.end(), 4677));
  SafeErase(t, find(t.begin(), t.end(), 4716));
  SafeErase(t, find(t.begin(), t.end(), 4715));
  SafeErase(t, find(t.begin(), t.end(), 4717));

  // accademia
  t.push_back(2528);
  t.push_back(2529);
  t.push_back(5459);
  t.push_back(2578);
  t.push_back(1285);
  t.push_back(5431);
  t.push_back(4277);

#else
  // zona accesso nord-ovest
  t.push_back(2);
  t.push_back(5095);
  t.push_back(5096);
  t.push_back(5098);
  t.push_back(5099);
  t.push_back(5100);
  t.push_back(5100);
  t.push_back(5177);
  t.push_back(6194);
  t.push_back(6197);
  t.push_back(6198);
  t.push_back(6219);
  t.push_back(6239);
  t.push_back(6240);
  t.push_back(5075);
  t.push_back(5357);
  t.push_back(5393);
  t.push_back(5394);
  t.push_back(5395);
  t.push_back(6177);
  t.push_back(6171);
  t.push_back(6168);

  // zona universita' sud-ovest
  t.push_back(3989);
  t.push_back(4024);
  t.push_back(4134);
  t.push_back(4792);
  t.push_back(5431);
  t.push_back(5854);
  t.push_back(5886);
  t.push_back(668);
  t.push_back(759);

  // zona cannaregio sud
  t.push_back(1201);
  t.push_back(1458);
  t.push_back(1459);
  t.push_back(1571);
  t.push_back(1572);
  t.push_back(1576);
  t.push_back(2443);
  t.push_back(2444);
  t.push_back(2445);
  t.push_back(3251);
  t.push_back(3252);
  t.push_back(3476);
  t.push_back(3482);
  t.push_back(3483);
  t.push_back(4007);
  t.push_back(4804);
  t.push_back(4878);
  t.push_back(4879);
  t.push_back(5741);
  t.push_back(5742);
  t.push_back(5754);
  t.push_back(5791);
  t.push_back(5987);
  t.push_back(614);
  t.push_back(755);
  t.push_back(937);
  t.push_back(1778);

  // zona centrale universita' di architettura
  t.push_back(1285);
  t.push_back(2578);
  t.push_back(3187);
  t.push_back(5448);
  t.push_back(5495);
  t.push_back(5502);
  t.push_back(5508);
  t.push_back(6133);
  t.push_back(6137);
  t.push_back(797);
  t.push_back(1777);

  // zona riva schiavoni sud-est
  t.push_back(134);
  t.push_back(2219);
  t.push_back(2220);
  t.push_back(2920);
  t.push_back(3135);
  t.push_back(3556);
  t.push_back(3557);
  t.push_back(4115);
  t.push_back(4143);
  t.push_back(5801);
  t.push_back(6144);
  SafeErase(t, find(t.begin(), t.end(), 4673));
  SafeErase(t, find(t.begin(), t.end(), 4674));
  SafeErase(t, find(t.begin(), t.end(), 4677));
  SafeErase(t, find(t.begin(), t.end(), 4717));

  // stazione
  t.push_back(4277);
  SafeErase(t, find(t.begin(), t.end(), 1056));
  SafeErase(t, find(t.begin(), t.end(), 1252));
  SafeErase(t, find(t.begin(), t.end(), 1011));
  SafeErase(t, find(t.begin(), t.end(), 1500));
  SafeErase(t, find(t.begin(), t.end(), 2545));
  SafeErase(t, find(t.begin(), t.end(), 3029));
  SafeErase(t, find(t.begin(), t.end(), 4288));
  SafeErase(t, find(t.begin(), t.end(), 4289));
  SafeErase(t, find(t.begin(), t.end(), 4290));
  SafeErase(t, find(t.begin(), t.end(), 5124));
  SafeErase(t, find(t.begin(), t.end(), 5412));
  SafeErase(t, find(t.begin(), t.end(), 5597));
  SafeErase(t, find(t.begin(), t.end(), 4397));
  SafeErase(t, find(t.begin(), t.end(), 5202));
  SafeErase(t, find(t.begin(), t.end(), 5183));

#endif

}
//----------------------------------------------------------------------------------------------------
template<typename T>
void CorreggiModello(T &t)
{
  // giudecca
  t.push_back(619);
  t.push_back(2453);
  t.push_back(2454);
  t.push_back(986);
  t.push_back(873);
  t.push_back(1340);
  SafeErase(t, find(t.begin(), t.end(), 2380));
  SafeErase(t, find(t.begin(), t.end(), 835));
  SafeErase(t, find(t.begin(), t.end(), 1338));
  SafeErase(t, find(t.begin(), t.end(), 6149));
  SafeErase(t, find(t.begin(), t.end(), 2458));
  SafeErase(t, find(t.begin(), t.end(), 4335));
  SafeErase(t, find(t.begin(), t.end(), 4334));
  SafeErase(t, find(t.begin(), t.end(), 4348));
  SafeErase(t, find(t.begin(), t.end(), 5132));
  SafeErase(t, find(t.begin(), t.end(), 5965));
  SafeErase(t, find(t.begin(), t.end(), 1725));
  SafeErase(t, find(t.begin(), t.end(), 4749));
  SafeErase(t, find(t.begin(), t.end(), 5634));
  SafeErase(t, find(t.begin(), t.end(), 5635));
  SafeErase(t, find(t.begin(), t.end(), 875));
  SafeErase(t, find(t.begin(), t.end(), 1835));
  SafeErase(t, find(t.begin(), t.end(), 1834));
  SafeErase(t, find(t.begin(), t.end(), 911));
  SafeErase(t, find(t.begin(), t.end(), 1074));
  SafeErase(t, find(t.begin(), t.end(), 3946));
  SafeErase(t, find(t.begin(), t.end(), 3945));
  SafeErase(t, find(t.begin(), t.end(), 1117));
  SafeErase(t, find(t.begin(), t.end(), 988));
  SafeErase(t, find(t.begin(), t.end(), 1029));
  SafeErase(t, find(t.begin(), t.end(), 3891));
  SafeErase(t, find(t.begin(), t.end(), 3892));
  SafeErase(t, find(t.begin(), t.end(), 5651));
  SafeErase(t, find(t.begin(), t.end(), 2500));
  SafeErase(t, find(t.begin(), t.end(), 2502));
  SafeErase(t, find(t.begin(), t.end(), 2503));

  // uni foscari
  SafeErase(t, find(t.begin(), t.end(), 5390));
  SafeErase(t, find(t.begin(), t.end(), 5075));
  SafeErase(t, find(t.begin(), t.end(), 5432));
  SafeErase(t, find(t.begin(), t.end(), 5431));
  SafeErase(t, find(t.begin(), t.end(), 4790));

  // darsena
  t.push_back(2184);
  t.push_back(2185);
  t.push_back(2183);
  t.push_back(4950);
  t.push_back(4951);
  t.push_back(2028);
  t.push_back(304);
  t.push_back(2034);
  t.push_back(4919);
  t.push_back(5332);
  t.push_back(4905);
  t.push_back(4906);
  t.push_back(4902);
  t.push_back(4920);
  t.push_back(209);
  t.push_back(2045);
  t.push_back(2669);
  t.push_back(2665);
  t.push_back(918);
  t.push_back(2167);
  t.push_back(2168);
  t.push_back(1420);
  t.push_back(1427);

}
void ImportaSubnet(const string &filename)
{
  ifstream sub(filename);
  if (!sub)
  {
    cout << "*** WARNING *** Unable to find input file : " << filename << endl;
    return;
  }
  else
    cout << "ImportaSubnet: importo subnet file " << filename << endl;

  subnets.clear();

  string line;
  vector<string> tokens;
  while (getline(sub, line))
  {
    physycom::split(tokens, line, string("\t"), physycom::token_compress_on);
    for (int i = 1; i < (int)tokens.size(); ++i) subnets[tokens[0]].push_back(stoi(tokens[i]));
  }

  //for (auto &s : subnets)
  //{
  //  if (s.first.find("_10") != string::npos)
  //    subnets["tutto_10"].insert(subnets["tutto_10"].end(), s.second.begin(), s.second.end());
  //  if (s.first.find("IT_10") != string::npos || s.first.find("ST_10") != string::npos)
  //    subnets["merge_10"].insert(subnets["merge_10"].end(), s.second.begin(), s.second.end());
  //  if (s.first.find("IT_15") != string::npos || s.first.find("ST_15") != string::npos)
  //    subnets["merge_15"].insert(subnets["merge_15"].end(), s.second.begin(), s.second.end());
  //  if (s.first.find("IT_20") != string::npos || s.first.find("ST_20") != string::npos)
  //    subnets["merge_20"].insert(subnets["merge_20"].end(), s.second.begin(), s.second.end());
  //}

  // uncomment to hack subnet
  //subnets["real_10"] = subnets["merge_10"]; HackeraSubnet(subnets["real_10"]);
  //subnets["real_15"] = subnets["merge_15"]; HackeraSubnet(subnets["real_15"]);
  //subnets["real_20"] = subnets["merge_20"]; HackeraSubnet(subnets["real_20"]);

  for (int i = 0; i < (int)nl_poly.size(); ++i)
  {
    if (i > 100) break;
    subnets["tutto_10"].push_back(nl_poly[i].first);
  }

  //HackeraSubnet(subnets["tutto_10"]);
  for (auto &p : subnets)
  {
    sort(p.second.begin(), p.second.end());
    p.second.erase(unique(p.second.begin(), p.second.end()), p.second.end());
  }

  subnets["corretta_10"] = subnets["tutto_10"];
  //CorreggiModello(subnets["corretta_10"]);
  //
}
//----------------------------------------------------------------------------------------------------
void CreaSubnet()
{
  // parse subnet file and
  // create menu button
  string pesidir = "../output/pesi";
  boost::filesystem::path targetDir(pesidir);
  if (! boost::filesystem::exists(targetDir))
  {
    cout << "CreaSubnet: non trovo path pesi " << pesidir << endl;
    return;
  }

  boost::filesystem::directory_iterator it(targetDir), eod;
  BOOST_FOREACH(boost::filesystem::path const &p, std::make_pair(it, eod))
  {
    if(boost::filesystem::is_regular_file(p))
    {
      if( p.string().find(".sub") == string::npos ) continue;
      string entry = "Import/Subnet/" + p.filename().string();
      menu->add(entry.c_str() , "", NULL);
    }
  }

  // setup gui subnet toggle
  offset_h += b_h;
  button32 = new Fl_Check_Button(space, offset_h, b_w, b_h, "Subnet"); offset_h += b_h;
  button32->callback([](Fl_Widget*)
  {
    set_off();
    button32->set();
    mostraSubnet = !mostraSubnet;
  });

  // setup callback to update list of
  // avalaible subnet to display
  menu->callback([](Fl_Widget* p)
  {
    char lbl[100];
    ((Fl_Sys_Menu_Bar*)p)->item_pathname(lbl, 100);
    string filename(lbl);
    filename = filename.substr(filename.find_last_of("/")+1);

    ImportaSubnet("../output/pesi/" + filename);
    subnet_label = "Subnet: \n" + filename.substr(filename.find_last_of("_")+1);
    button32->label(subnet_label.c_str());
    offset_h = button32->y() + int(1.5*b_h);

    if( subnet_buttons.size() )
    {
      for(const auto sb : subnet_buttons)
      {
        form->remove(form->find(sb));
      }
      subnet_buttons.clear();
    }

    for (const auto &p : subnets)
    {
      Fl_Check_Button* o = new Fl_Check_Button(int(1.2*space), offset_h, b_w, int(0.5*b_h), p.first.c_str()); offset_h += int(0.5*b_h);
      o->labelsize(SMALL_FONT_SIZE);
      o->type(FL_RADIO_BUTTON);
      o->callback([](Fl_Widget*, void* ptr)
      {
        button32->callback()(nullptr, nullptr);
        subnet_select = ((pair<string, vector<int>>*)ptr)->first;
        string message = "Subnet : '" + subnet_select + "' (" + to_string(subnets[subnet_select].size()) +
          ") " + to_string(int(CalcolaSubnet(subnet_select) * 100)) + "% length @ " +
          to_string(int(CalcolaRappresentanza_media(subnet_select) * 100)) + "% mobility";
        linea2->value(message.c_str());
        reDraw = true;
      }, (void*)&p);
      form->add(o);
      subnet_buttons.push_back(o);
    }
    form->redraw();
  });
}
//----------------------------------------------------------------------------------------------------
void draw_subnet()
{
  color_palette(2);
  glLineWidth(3.0);
  glPushMatrix();
  for (const auto &n : subnets[subnet_select]) {
    glBegin(GL_LINE_STRIP);
    for (const auto &j : polyline[n].points) glVertex3d(j.lon, j.lat, 0.3);
    glEnd();
  }

  glPopMatrix();
  glLineWidth(1.0);
  glColor3d(1.0, 1.0, 1.0);
}
//----------------------------------------------------------------------------------------------------
