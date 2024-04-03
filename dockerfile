#    Serves static files and reloads the page on changes to them
#
#    Copyright (C) 2024 Till Blaha -- TU Delft
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# build and invoke with 
#   docker build -t devserver . -f dockerfile
#   docker run --net host -v ./your/static/files:/usr/app/static devserver

FROM node:alpine

LABEL maintainer="t.m.blaha@tudelft.nl"
LABEL version="0.1"
LABEL description="Serves static files and reloads the page on changes to them"

WORKDIR /usr/
RUN npm create vite app -- --template vanilla

WORKDIR /usr/app
RUN npm install three
RUN npm install stats.js

RUN echo "export default { root:'static' }" > /usr/app/vite.config.js

EXPOSE 5173

CMD [ "npm", "run", "dev" ]
