AC_INIT([Opencal-gl], [0.1], [donato.dambrosio@gmail.com])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CC
AM_PROG_AR
LT_INIT
AC_PROG_LIBTOOL

AC_CONFIG_HEADERS([config.h])

AC_ARG_WITH([opencal],
    [AS_HELP_STRING([--with-opencal=prefix],
        [try this for a non-standard install prefix of the opencal library])],
    [opencal_pathset=1],
    [opencal_pathset=0])

AM_CONDITIONAL([OPENCAL_PATH], [test $opencal_pathset = 1])

if test $opencal_pathset = 1; then
   case "${with_opencal}" in
      /*) oc_dir=${with_opencal} ;;
      *) oc_dir=$(pwd)/${with_opencal} ;;
   esac  
   OPENCAL_CPPFLAGS="-I${oc_dir}/include"
   OPENCAL_LDFLAGS="-L${oc_dir}/lib"
fi

CPPFLAGS="$OPENCAL_CPPFLAGS $CPPFLAGS"
LDFLAGS="$OPENCAL_LDFLAGS $LDFLAGS"


AC_CHECK_HEADERS([cal2D.h],
	[FOUND_OPENCAL=1],
	[FOUND_OPENCAL=0])

if test $FOUND_OPENCAL != 1; then
    AC_MSG_NOTICE([])
    AC_MSG_NOTICE([The Opencal library was not found!])
    if test  $opencal_pathset = 1 ; then
        AC_MSG_NOTICE([ The install prefix '$with_opencal' for the opencal library was set.])
        AC_MSG_NOTICE([ --> Maybe wrong ???])
    else
        AC_MSG_NOTICE([ No non-standard install prefix was set.])
        AC_MSG_NOTICE([ --> You might want to use '--with-opencal=PREFIX' ?!?])
    fi
    AC_MSG_NOTICE([])
    AC_MSG_ERROR([opencal library is an essential dependency : cannot build and stop here !])
fi


AC_CONFIG_FILES([
	Makefile
	source/Makefile
])
AC_OUTPUT
