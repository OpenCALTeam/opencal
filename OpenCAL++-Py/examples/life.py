from opencal import *
import opencal


Q = None
Qzito = None

class Life_transition_function(opencal.ElementaryProcessFunctor2D):
	def run(SELF,life,i,j):
			global Q
			s=0
			neighbor = range(1,life.sizeof_X)
			if not(Q is None):
				for n in neighbor:
					s =s+ opencal.calGetX2Di(life, Q, i, j, n);
					print 'cella',str(i),',',str(j),',vic',str(n),' val=',str(opencal.calGetX2Di(life, Q, i, j, n))
				if (s==3) or (s==2 and opencal.calGet2Di(life, Q, i, j) == 1):
						opencal.calSet2Di(life, Q, i, j,1)
				else:
						opencal.calSet2Di(life, Q, i, j,0)




def test():
	life2D = opencal.calCADef2D (10, 10, opencal.CAL_MOORE_NEIGHBORHOOD_2D, opencal.CAL_SPACE_TOROIDAL, opencal.CAL_NO_OPT);
	calAddElementaryProcess2D(life2D, Functor());


def life():



		life2D = opencal.calCADef2D (10, 10, opencal.CAL_MOORE_NEIGHBORHOOD_2D, opencal.CAL_SPACE_TOROIDAL, opencal.CAL_NO_OPT)

		print("START function life")
		print(life2D.columns)
		print(life2D.rows)

		life_simulation = opencal.calRunDef2D(life2D, 1, 6, opencal.CAL_UPDATE_EXPLICIT)


		life_transition_function = Life_transition_function()
		life_transition_function.run(life2D,1,1)

		opencal.calAddElementaryProcess2D(life2D, life_transition_function)
		#opencal.calAddElementaryProcess2D(life2D, life_elementary_process_1)
		#opencal.calAddElementaryProcess2D(life2D, life_elementary_process_2)



		#add substates
		global Q
		Q = opencal.calAddSubstate2Di(life2D);
		print(Q)
		global Qzito
		Qzito = opencal.calAddSubstate2Di(life2D);

		calInitSubstate2Di(life2D, Q, 0);
		calInitSubstate2Di(life2D, Qzito, 0);

		#set a glider
		ri=2
		ci=2
		opencal.calInit2Di(life2D, Q, 0+ri, 2+ci, 1);
		opencal.calInit2Di(life2D, Q, 1+ri, 0+ci, 1);
		opencal.calInit2Di(life2D, Q, 1+ri, 2+ci, 1);
		opencal.calInit2Di(life2D, Q, 2+ri, 1+ci, 1);
		opencal.calInit2Di(life2D, Q, 2+ri, 2+ci, 1);

		#saving configuration
		opencal.calSaveSubstate2Di(life2D, Q, "./life_0000.txt");
		opencal.calSaveSubstate2Di(life2D, Qzito, "./zito_0000.txt");


		#simulation run
		opencal.calRun2D(life_simulation);
		#opencal.calRunFinalize2D(life_simulation);

		opencal.calSaveSubstate2Di(life2D, Q, "./life_0000.txtEND");
		opencal.calSaveSubstate2Di(life2D, Qzito, "./zito_0000.txtEND");



life()
print("END")
