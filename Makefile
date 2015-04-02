include Makefile.inc

OPENCAL_PATH = ./OpenCAL/
ISO3SRC_PATH = ./iso-3src-glut/
ISO_PATH = ./iso-glut/
LIFE_PATH = ./life/
LIFE3DGLUT_PATH = ./life3D-glut/
MBUSU_PATH = ./mbusu-glut
SCIDDICAS3HEX_PATH = ./sciddicaS3hex-unsafe-glut/
SCIDDICAT_PATH = ./sciddicaT/
SCIDDICATACTIVECELLSGLUT_PATH = ./sciddicaT-activecells-glut/
SCIDDICATGLUT_PATH = ./sciddicaT-glut/
SCIDDICATUNSAFEGLUT_PATH = ./sciddicaT-unsafe-glut/
SCIDDICATUNSAFE_PATH = ./sciddicaT-unsafe/
#BENCHMARK_PATH = ./benchmark



all:
	cd $(OPENCAL_PATH) && $(MAKE)
	cd $(ISO3SRC_PATH) && $(MAKE)
	cd $(ISO_PATH) && $(MAKE)
	cd $(LIFE_PATH) && $(MAKE)
	cd $(LIFE3DGLUT_PATH) && $(MAKE)
	cd $(MBUSU_PATH) && $(MAKE)
	cd $(SCIDDICAS3HEX_PATH) && $(MAKE)
	cd $(SCIDDICAT_PATH) && $(MAKE)
	cd $(SCIDDICATACTIVECELLSGLUT_PATH) && $(MAKE)
	cd $(SCIDDICATGLUT_PATH) && $(MAKE)
	cd $(SCIDDICATUNSAFEGLUT_PATH) && $(MAKE)
	cd $(SCIDDICATUNSAFE_PATH) && $(MAKE)
#	cd $(BENCHMARK_PATH) && $(MAKE) clean	

clean:
	cd $(OPENCAL_PATH) && $(MAKE) clean
	cd $(ISO3SRC_PATH) && $(MAKE) clean
	cd $(ISO_PATH) && $(MAKE) clean
	cd $(LIFE_PATH) && $(MAKE) clean
	cd $(LIFE3DGLUT_PATH) && $(MAKE) clean
	cd $(MBUSU_PATH) && $(MAKE) clean
	cd $(SCIDDICAS3HEX_PATH) && $(MAKE) clean
	cd $(SCIDDICAT_PATH) && $(MAKE) clean
	cd $(SCIDDICATACTIVECELLSGLUT_PATH) && $(MAKE) clean
	cd $(SCIDDICATGLUT_PATH) && $(MAKE) clean
	cd $(SCIDDICATUNSAFEGLUT_PATH) && $(MAKE) clean
	cd $(SCIDDICATUNSAFE_PATH) && $(MAKE) clean
#	cd $(BENCHMARK_PATH) && $(MAKE) clean	

#bench: 
#	cd $(OPENCAL_PATH) && $(MAKE) 
#	cd $(BENCHMARK_PATH) && $(MAKE)




