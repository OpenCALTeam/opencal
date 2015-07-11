// (C) Copyright University of Calabria and others.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the GNU Lesser General Public License
// (LGPL) version 2.1 which accompanies this distribution, and is available at
// http://www.gnu.org/licenses/lgpl-2.1.html
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.

#include <ElementaryProcessFunctor3D.h>

ElementaryProcessFunctor3D::ElementaryProcessFunctor3D() {

}

ElementaryProcessFunctor3D::~ElementaryProcessFunctor3D() {

}

void ElementaryProcessFunctor3D::operator()(CALModel3D* model, int i, int j,int k){
	this->run(model,i,j,k);
}
