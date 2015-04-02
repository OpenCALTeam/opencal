AC_INIT([OpenCal], [0.1], [donato.dambrosio@gmail.com])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CC
AM_PROG_AR
LT_INIT
AC_PROG_LIBTOOL

AC_CONFIG_HEADERS([config.h])

AC_ARG_ENABLE([openmp],
	[AC_HELP_STRING([--enable-openmp],
	[enabling multi-threading compution trought openmp])],
	[if test x"$enableval" = x"yes"; then
	    openmp=true
	 else
	    openmp=false
         fi],
	[openmp=false])

if test x"$openmp" = x"true"; then
   AX_OPENMP([openmp=true], [openmp=false])
fi

if test x"$openmp" = x"true"; then
   AC_DEFINE([CAL_OMP], [1], [Enabling openmp])
   CFLAGS="$CFLAGS $OPENMP_CFLAGS"
fi

AC_CONFIG_FILES([
	Makefile
	source/Makefile
])
AC_OUTPUT
